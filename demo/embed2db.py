import torch
import numpy as np

from config import config
from image_caption_dataset import ImageCaptionDataset
from db_handler import MilvusHandler

from tqdm import tqdm
from torch.utils.data import DataLoader
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel

from log_handler import Logger

logger = Logger.get_logger()


def create_milvus_collection(collection_name, dim):
    connections.connect(host=config.milvus.host, port=config.milvus.port)

    if utility.has_collection(collection_name):
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

    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


if __name__ == '__main__':

    checkpoint_dir = config.finetune.checkpoint_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'device: {device}')

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        checkpoint_dir).to(device)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        checkpoint_dir).to(device)
    model = CLIPModel.from_pretrained(checkpoint_dir).to(device)

    processor = AutoProcessor.from_pretrained(checkpoint_dir)

    dataset = ImageCaptionDataset(
        is_train=config.dataset.is_train, return_loss=False, dataset=config.dataset.name)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,  # iter-style dataset set 0
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        collate_fn=dataset.collate_fn
    )

    milvus_handler = MilvusHandler()
    milvus_handler.create_collection(
        config.milvus.collection_name, config.milvus.vector_dim)
    milvus_handler._connect_collection(config.milvus.collection_name)

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            ids, categories, inputs = batch

            if inputs['pixel_values'].shape[0] != inputs['input_ids'].shape[0]:
                len_list = inputs['len_list']
                del inputs['len_list']

                inputs = inputs.to(device)

                image_output = image_encoder(
                    pixel_values=inputs['pixel_values'])
                image_output.image_embeds /= image_output.image_embeds.norm(
                    dim=-1, keepdim=True)
                image_embeds = image_output.image_embeds.squeeze().cpu().numpy()

                text_output = text_encoder(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                text_output.text_embeds /= text_output.text_embeds.norm(
                    dim=-1, keepdim=True)
                text_embeds = text_output.text_embeds.squeeze().cpu().numpy()

                text_embeds_list = []
                for i in range(len(len_list)):
                    sub_text_embeds = text_embeds[sum(
                        len_list[:i]):sum(len_list[:i]) + len_list[i]]

                    sub_text_embeds = np.mean(
                        sub_text_embeds, axis=0, keepdims=True)
                    text_embeds_list.append(sub_text_embeds)

                mean_text_embeds = np.concatenate(text_embeds_list, axis=0)

                insert_datas = [
                    ids, (image_embeds + mean_text_embeds) / 2, categories]
                mr = milvus_handler.insert(data=insert_datas)
                continue

            output = model(**inputs)

            output.image_embeds /= output.image_embeds.norm(
                dim=-1, keepdim=True)
            image_embeds = output.image_embeds.squeeze().cpu().numpy()

            output.text_embeds /= output.text_embeds.norm(dim=-1, keepdim=True)
            text_embeds = output.text_embeds.squeeze().cpu().numpy()

            insert_datas = [ids, (image_embeds + text_embeds) / 2, categories]

            mr = milvus_handler.insert(data=insert_datas)

    milvus_handler.load_and_flush()
    milvus_handler.get_num_entities()
    milvus_handler.list_collections()
