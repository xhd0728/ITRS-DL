import os
import yaml
import torch
import onnxruntime
import numpy as np
from config import config
from PIL import Image
from transformers import AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, ChineseCLIPModel, ChineseCLIPProcessor


class OnnxTextModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.onnx_path = os.path.join(
            config.onnx.save_dir, model_name.split('/')[-1] + '_text_encoder.onnx')
        self.providers = 'CUDAExecutionProvider' if torch.cuda.is_available(
        ) else 'CPUExecutionProvider'
        self.session = onnxruntime.InferenceSession(
            self.onnx_path, providers=[self.providers])
        self.processor = AutoProcessor.from_pretrained(model_name)

    def __call__(self, text):
        text = self.processor(text=text, return_tensors='np', padding=True)
        text_token = dict(text)
        for i in text_token:
            text_token[i] = text_token[i].astype(np.int64)
        text_embeds = self.session.run(None, text_token)[0]
        return text_embeds / np.linalg.norm(text_embeds, axis=1, keepdims=True)


class HfTextModel:
    def __init__(self, model_name):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.processor = self._load_model_and_processor(model_name)
        self._warmup(model_name)

    def _load_model_and_processor(self, model_name):
        if 'chinese' in model_name:
            model = ChineseCLIPModel.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            ).to(self.device)
            processor = ChineseCLIPProcessor.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            )
        else:
            model = CLIPTextModelWithProjection.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    def _warmup(self, model_name):
        input = self.processor(
            text='warmup text',
            return_tensors='pt',
            padding=True
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if 'chinese' in model_name:
                self.model.get_text_features(**input)
            else:
                self.model(**input)

    @classmethod
    def _empty_cache(self):
        torch.cuda.empty_cache()

    def _reload(self, model_name):
        self._empty_cache()
        self.model_name = model_name
        self.model, self.processor = self._load_model_and_processor(model_name)

    def __call__(self, text):
        text_token = self.processor(
            text=text,
            return_tensors='pt',
            padding=True
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_token)

        text_features /= text_features.norm(
            dim=-1,
            keepdim=True
        )
        text_embeds = text_features.cpu().numpy()
        return text_embeds


class HfVisionModel:
    def __init__(self, model_name):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.processor = self._load_model_and_processor(model_name)
        self._warmup(model_name)

    def _load_model_and_processor(self, model_name):
        if 'chinese' in model_name:
            model = ChineseCLIPModel.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            ).to(self.device)
            processor = ChineseCLIPProcessor.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            )
        else:
            model = CLIPVisionModelWithProjection.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(model_name)
        return model, processor

    def _warmup(self, model_name):
        # input = torch.randn(1, 3, 224, 224).to(self.device)
        input = self.processor(
            images=Image.new("RGB", (30, 30), (255, 255, 255)),
            return_tensors='pt',
            padding=True
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if 'chinese' in model_name:
                self.model.get_image_features(**input)
            else:
                self.model(input)

    @classmethod
    def _empty_cache(self):
        torch.cuda.empty_cache()

    def _reload(self, model_name):
        self._empty_cache()
        self.model_name = model_name
        self.model, self.processor = self._load_model_and_processor(model_name)

    def __call__(self, images):
        image_token = self.processor(
            images=images, return_tensors='pt', padding=True).to(self.device)
        self.model.eval()
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_token)

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )
        image_embeds = image_features.cpu().numpy()
        return image_embeds
