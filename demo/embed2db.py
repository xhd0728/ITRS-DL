import os
import json
import yaml

import torch
import pandas as pd
import numpy as np

from config import config
from image_caption_dataset import ImageCaptionDataset
from db_handler import MilvusHandler

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility, db
from transformers import AutoModel, AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel


def create_milvus_collection(collection_name, dim):
    connections.connect(host=config.milvus.host, port=config.milvus.port)

    if utility.has_collection(collection_name):  # 如果数据库存在则删除
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64,
                    descrition='ids', is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                    descrition='embedding vectors', dim=dim),
        FieldSchema(name='category', dtype=DataType.INT64,
                    descrition='category'),
    ]

    schema = CollectionSchema(
        fields=fields, description='mini imagenet text image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = config.milvus.index_params

    # 根据字段建立索引
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


if __name__ == '__main__':
    # 权重路径
    checkpoint_dir = config.finetune.checkpoint_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    # 模型加载权重
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        checkpoint_dir).to(device)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        checkpoint_dir).to(device)
    model = CLIPModel.from_pretrained(checkpoint_dir).to(device)
    # 预处理器
    processor = AutoProcessor.from_pretrained(checkpoint_dir)

    dataset = ImageCaptionDataset(
        is_train=False, return_loss=False, dataset=config.dataset.name)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,  # iter-style dataset 需要设置为0
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    milvus_handler = MilvusHandler()
    milvus_handler.create_collection(
        config.milvus.collection_name, config.milvus.vector_dim)
    milvus_handler._connect_collection(config.milvus.collection_name)

    # 调用显卡inference
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            ids, categories, inputs = batch

            # 如果图像数量与描述数量不一样, 即如果文对图是多对一
            if inputs['pixel_values'].shape[0] != inputs['input_ids'].shape[0]:
                len_list = inputs['len_list']
                del inputs['len_list']

                # debug
                inputs = inputs.to(device)

                # 图像部分
                image_output = image_encoder(
                    pixel_values=inputs['pixel_values'])
                image_output.image_embeds /= image_output.image_embeds.norm(
                    dim=-1, keepdim=True)
                image_embeds = image_output.image_embeds.squeeze().cpu().numpy()

                # 文本部分
                text_output = text_encoder(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                text_output.text_embeds /= text_output.text_embeds.norm(
                    dim=-1, keepdim=True)
                text_embeds = text_output.text_embeds.squeeze().cpu().numpy()

                # 对batch中每个图像对应的多个描述做平均embed
                text_embeds_list = []
                for i in range(len(len_list)):
                    sub_text_embeds = text_embeds[sum(
                        len_list[:i]):sum(len_list[:i]) + len_list[i]]
                    # 将多个描述的text_embedding平均
                    sub_text_embeds = np.mean(
                        sub_text_embeds, axis=0, keepdims=True)
                    text_embeds_list.append(sub_text_embeds)
                # 将分散的text_embed组成一个batch
                mean_text_embeds = np.concatenate(text_embeds_list, axis=0)
                # 可以传list[dict] or list[list]
                insert_datas = [
                    ids, (image_embeds + mean_text_embeds) / 2, categories]
                mr = milvus_handler.insert(data=insert_datas)
                continue

            # inputs = inputs.to(device)
            output = model(**inputs)

            # 图像部分
            output.image_embeds /= output.image_embeds.norm(
                dim=-1, keepdim=True)
            image_embeds = output.image_embeds.squeeze().cpu().numpy()  # 注意要squeeze() 不然无法插入

            # 文本部分
            output.text_embeds /= output.text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = output.text_embeds.squeeze().cpu().numpy()

            # 可以传list[dict] or list[list]
            insert_datas = [ids, (image_embeds + text_embeds) / 2, categories]
            # mr = collection.insert(data=insert_datas)
            mr = milvus_handler.insert(data=insert_datas)

    milvus_handler.load_and_flush()  # 加载向量数据到内存中
    milvus_handler.get_num_entities()
    milvus_handler.list_collections()
