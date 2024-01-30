import warnings
import os
import io
import pickle
import base64

import gradio as gr
import numpy as np
import pandas as pd

from config import config
from model import OnnxTextModel, HfTextModel, HfVisionModel
from image_caption_dataset import get_labels_and_cates
from db_handler import MilvusHandler, RedisHandler
from metric import NDCG, MRR, mAP

from PIL import Image
from log_handler import Logger
logger = Logger.get_logger()


warnings.filterwarnings("ignore")


labels, cates = get_labels_and_cates(
    dataset=config.dataset.name,
    is_train=config.dataset.is_train
)
label2cate = {label: cate for label, cate in zip(labels, cates)}
cate2label = {str(cate): label for label, cate in zip(labels, cates)}

extraPairDict = {}


def id2image(img_id: int) -> Image.Image:

    root_dir = config.dataset.flickr30k_path
    val_data = pd.read_csv(os.path.join(root_dir, 'flickr30k.csv'))

    if img_id >= len(val_data['filename']):
        img_base64 = extraPairDict[str(img_id)]
        img = Image.open(io.BytesIO(
            base64.b64decode(img_base64))).convert("RGB")
    else:
        img_path = os.path.join(
            root_dir, 'flickr30k-images', val_data['filename'][img_id])
        img = Image.open(img_path)
    return img


class QueryService:
    def __init__(self, model_name, use_onnx=config.onnx.use_onnx, use_redis=config.redis.use_redis):
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.use_redis = use_redis

        if self.use_redis:
            self.redis_handler = RedisHandler()

        self.milvus_handler = MilvusHandler()
        self.text_model = OnnxTextModel(
            model_name) if self.use_onnx else HfTextModel(model_name)
        self.vision_model = HfVisionModel(model_name)

    def __call__(self, query_text, topk, model_name):
        try:
            ids, _, categories = self._search_categories(query_text, topk)

            images = list(map(id2image, ids))

            captions = list(map(lambda x: cate2label[str(x)], categories))
        except Exception as e:
            logger.error(f'redis error: {e}')
            return self.__call__(query_text, topk, model_name)
        return list(zip(images, captions))

    def _search_categories(self, query_text, topk):
        if self.use_redis:

            if isinstance(query_text, str):
                search_res = self.redis_handler.get(query_text)

            elif isinstance(query_text, list):
                search_res = self.redis_handler.redis_client.mget(query_text)
            else:
                return NotImplementedError

            if search_res is None or None in search_res:
                ids, distances, categories = self._embed_and_search(
                    query_text, topk)

                res_pack = list(zip(ids, distances, categories))

                if isinstance(query_text, str):
                    query_text = [query_text]
                for text, pack in zip(query_text, res_pack):
                    self.redis_handler.set(text, pickle.dumps(pack))

                if not isinstance(query_text, str):
                    return ids, distances, categories
                else:
                    return ids[0], distances[0], categories[0]
            else:
                if isinstance(query_text, str):
                    id, distance, category = self.redis_handler.get(query_text)
                    return id, distance, category
                elif isinstance(query_text, list):
                    deserialize_res = self.redis_handler.mget(query_text)
                    ids, distances, categories = tuple(zip(*deserialize_res))
                    return ids, distances, categories

        else:
            ids, distances, categories = self._embed_and_search(
                query_text, topk)
            if not isinstance(query_text, str):
                return ids, distances, categories
            else:
                return ids[0], distances[0], categories[0]

    def _embed_and_search(self, query_text, topk):
        text_embeds = self.text_model(text=query_text)
        ids, distances, categories = self.milvus_handler.search(
            text_embeds, topk)
        return ids, distances, categories

    def _PIL2Base64(self, image):

        image_data = io.BytesIO()

        image.save(image_data, format='JPEG')

        image_data_bytes = image_data.getvalue()

        encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
        return encoded_image

    def embed_and_insert(self, upload_image, label):
        image_embeds = self.vision_model(images=upload_image)
        text_embeds = self.text_model(text=label)
        ids = [self.milvus_handler.collection.num_entities + 1]

        if label2cate.get(label) is None:
            labels.append(label)
            label2cate[label] = max(cates) + 1
            cates.append(label2cate[label])
            cate2label[str(label2cate[label])] = label
            categories = [label2cate[label]]

            extraPairDict[str(
                self.milvus_handler.collection.num_entities + 1)] = self._PIL2Base64(upload_image)
        else:
            categories = [label2cate[label]]

        insert_datas = [ids, (image_embeds + text_embeds) / 2, categories]
        self.milvus_handler.insert(data=insert_datas)
        return 'success'

    def compute_metrics(self, query_text=labels):
        from sklearn.metrics import recall_score
        recalls = []
        mrrs = []
        ndcgs = []
        maps = []

        topk_list = [1, 3, 5, 10]
        ids, _, categories = self._search_categories(
            query_text, max(topk_list))

        for k in topk_list:

            targets = np.array(cates)
            categories_k = np.array(categories)[:, :k]

            targets_repeat = targets.repeat(k)
            categories_flat = categories_k.flatten()

            recall = recall_score(
                targets_repeat, categories_flat, average='micro')

            mrr = MRR(categories_k, targets)
            ndcg = NDCG(categories_k, targets)
            m_ap = mAP(categories_k, targets)

            recalls.append(round(100 * recall, 4))

            mrrs.append(round(100 * mrr, 4))
            ndcgs.append(round(100 * ndcg, 4))
            maps.append(round(100 * m_ap, 4))

        return f"""
                |            | **Recall (%)** | **MRR (%)** | **NDCG (%)** | **mAP (%)** |
                |:----------:|:--------------:|:-----------:|:------------:|:-----------:|
                |  **top@1** |{recalls[0]}    |{mrrs[0]}    |{ndcgs[0]}    |{maps[0]}    |
                |  **top@3** |{recalls[1]}    |{mrrs[1]}    |{ndcgs[1]}    |{maps[1]}    |
                |  **top@5** |{recalls[2]}    |{mrrs[2]}    |{ndcgs[2]}    |{maps[2]}    |
                | **top@10** |{recalls[3]}    |{mrrs[3]}    |{ndcgs[3]}    |{maps[3]}    |
                """


def text2image_gr(model_query, model_name=config.gradio.checkpoint_dir):
    clip = model_name

    title = "<h1 align='center'>ITRS_DL</h1>"
    description = "<h3 align='center'>基于深度学习的图文检索系统</h3>"

    examples = [
        ["dog", 10, clip],
    ]

    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(scale=2):
                    query_text = gr.Textbox(
                        value="dog", label="input to search...", elem_id=0, interactive=True)

                topk = gr.components.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=10,
                    label="top_k",
                    elem_id=2
                )

                model_name = gr.components.Radio(
                    label="model_select",
                    choices=[clip],
                    value=clip, elem_id=3
                )

                btn1 = gr.Button("retrieval")

            with gr.Column(scale=100):
                out1 = gr.Gallery(label="result:", columns=5, height=350)

            with gr.Column(scale=2):
                with gr.Column(scale=6):
                    out2 = gr.Markdown(
                        """
                        |            | **Recall (%)** | **MRR (%)** | **NDCG (%)** | **mAP (%)** |
                        |:----------:|:--------------:|:-----------:|:------------:|:-----------:|
                        |  **top@1** |                |             |              |             |
                        |  **top@3** |                |             |              |             |
                        |  **top@5** |                |             |              |             |
                        | **top@10** |                |             |              |             |
                        """
                    )
                btn2 = gr.Button("calc params", scale=1)

        inputs = [query_text, topk, model_name]

        gr.Examples(examples, inputs=inputs)

        btn1.click(fn=model_query, inputs=inputs, outputs=out1)
        btn2.click(fn=model_query.compute_metrics, inputs=None, outputs=out2)

    return demo


def upload2db_gr(model_query):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img = gr.Image(type='pil')
                label = gr.Textbox(label='category')
                with gr.Row():
                    gr.ClearButton(img)
                    btn = gr.Button("submit")

            with gr.Column():
                md = gr.Markdown()

        btn.click(fn=model_query.embed_and_insert,
                  inputs=[img, label], outputs=md)
    return demo


if __name__ == "__main__":
    model_name = config.gradio.checkpoint_dir
    model_query = QueryService(model_name)

    with gr.TabbedInterface(
            [text2image_gr(model_query, model_name),
             upload2db_gr(model_query)],
            ['query', 'upload'],
    ) as demo:
        demo.launch(
            enable_queue=True,
            server_name='0.0.0.0'
        )
