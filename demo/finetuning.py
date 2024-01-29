# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 6.py --ep=5

from log_handler import Logger
import os
import json
import yaml
import requests

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from config import config
from image_caption_dataset import ImageCaptionDataset
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, SubsetRandomSampler
from torch.nn import CrossEntropyLoss, NLLLoss

from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoModel
from transformers import Trainer, TrainingArguments

from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


logger = Logger.get_logger()


class DifferentClassSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        class_indices = {}
        for i, (_, _, _, caption) in enumerate(dataset):
            if caption not in class_indices:
                class_indices[caption] = []
            class_indices[caption].append(i)
        self.class_indices = class_indices

    def __iter__(self):
        indices = []
        for caption in self.class_indices.keys():
            class_indices = self.class_indices[caption]
            class_indices = list(class_indices)
            indices.append(class_indices)

        max_length = max(len(lst) for lst in indices)
        padded_lists = [lst + [None] * (max_length - len(lst))
                        for lst in indices]
        merged_indices = [item for sublist in zip(
            *padded_lists) for item in sublist if item is not None]
        return iter(merged_indices)

    def __len__(self):
        return len(self.dataset)


def train_fn(args, model, optimizer, dataloader):
    model.train()
    train_loss = []
    for batch in tqdm(dataloader, disable=not args.accelerator.is_main_process):
        optimizer.zero_grad()
        _, _, inputs = batch
        loss = model(**inputs).loss
        args.accelerator.backward(loss)
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)


@torch.no_grad()
def predict_fn(args, model, dataloader):

    model.eval()
    val_loss, val_acc, num_examples = 0, 0, 0

    for step, batch in enumerate(tqdm(dataloader, disable=not args.accelerator.is_main_process)):

        _, _, inputs = batch
        output = model(**inputs)

        val_loss += output.loss.item()

        logits_per_image = output.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predictions = torch.argmax(probs, dim=-1)
        predictions = args.accelerator.gather_for_metrics(predictions)

        labels = torch.arange(len(predictions)).to(args.accelerator.device)
        accuracy = torch.sum(predictions == labels)
        num_examples += len(predictions)
        val_acc += accuracy

    val_acc = val_acc.detach().cpu().numpy() / num_examples
    return val_acc


def fit_model(args, model, optimizer, train_dl, val_dl):
    best_score = 0.
    low_loss = 999999.
    for ep in range(args.ep):
        train_loss = train_fn(args, model, optimizer, train_dl)
        val_acc = predict_fn(args, model, val_dl)
        if args.accelerator.is_main_process:
            logger.info(
                f"Epoch {ep + 1}, train loss {train_loss:.4f}, val acc {val_acc:.4f}")

        if train_loss < low_loss:
            low_loss = train_loss

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                config.finetune.save_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )

    model = AutoModel.from_pretrained(config.finetune.save_dir)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", default=5, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--bs", default=32, type=int)
    args = parser.parse_args()

    train_ds = ImageCaptionDataset(
        is_train=True, return_loss=True, dataset=config.dataset.name)
    val_ds = ImageCaptionDataset(
        is_train=False, return_loss=True, dataset=config.dataset.name)

    train_diffsampler = DifferentClassSampler(train_ds)
    val_diffsampler = DifferentClassSampler(val_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.bs,
        num_workers=4,
        shuffle=False,
        # sampler=train_diffsampler,
        pin_memory=False,
        collate_fn=train_ds.collate_fn,
        drop_last=False)

    val_dl = DataLoader(
        val_ds,
        batch_size=args.bs,
        num_workers=4,
        shuffle=False,
        sampler=val_diffsampler,
        pin_memory=False,
        collate_fn=val_ds.collate_fn,
        drop_last=False)

    model = CLIPModel.from_pretrained(config.finetune.checkpoint_dir)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(
        0.9, 0.98), eps=1e-6, weight_decay=0.2)

    os.makedirs(config.finetune.save_dir, exist_ok=True)
    kwargs = InitProcessGroupKwargs(
        backend='gloo', timeout=timedelta(days=1))
    accelerator = Accelerator(
        kwargs_handlers=[kwargs], project_dir=config.finetune.save_dir)

    model, optimizer, train_dl, val_dl, = accelerator.prepare(
        model, optimizer, train_dl, val_dl)
    args.accelerator = accelerator

    model = fit_model(args, model, optimizer, train_dl, val_dl)
    logger.info(predict_fn(args, model, val_dl))
