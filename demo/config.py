model_list = [
    'clip-vit-base-patch16',
    'clip-vit-base-patch32',
    'clip-vit-large-patch14',
    'chinese-clip-vit-base-patch16',
    'chinese-clip-vit-large-patch14-336px'
]

model_name = model_list[4]


class MilvusConfig:
    def __init__(self):
        # self.host = '127.0.0.1'
        self.host = '8.217.103.200'
        self.port = '19530'
        self.collection_name = 'cn_vit_l_p14_flickr30k'
        self.vector_dim = 768
        self.topk = 10

        self.index_params = {
            'metric_type': 'IP',
            'index_type': 'HNSW',
            'params': {
                'M': 8,
                'efConstruction': 128
            }
        }

        self.search_params = {
            "metric_type": 'IP',
            "params": {
                "ef": 20
            }
        }


class RedisConfig:
    def __init__(self):
        self.use_redis = True
        self.host = '127.0.0.1'
        self.port = 6379
        self.db = 0
        self.expire_time = 3600  # 过期时间


class GradioConfig:
    def __init__(self):

        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'


class FinetuneConfig:
    def __init__(self):

        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'
        self.save_dir = './checkpoint'


class OnnxConfig:
    def __init__(self):
        self.use_onnx = False

        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'
        self.save_dir = './onnx'


dataset_list = [
    'mini-imagenet',
    'flickr30k',
    'flickr30k-cn',
    'coco2014',
    'coco2017',
]


class DatasetConfig:
    def __init__(self):

        self.name = dataset_list[2]

        self.mini_imagenet_path = '/mnt/f/DLWorks/dataset/mini-imagenet'
        self.flickr30k_path = '/mnt/f/DLWorks/dataset/flickr30k'
        self.flickr30k_cn_path = '/mnt/f/DLWorks/dataset/flickr30k-cn'
        self.coco_path = '/mnt/f/DLWorks/dataset/mscoco2014'
        self.coco2017_path = '/mnt/f/DLWorks/dataset/mscoco2017'

        self.is_train = False


class Config:
    def __init__(self):
        self.milvus = MilvusConfig()
        self.redis = RedisConfig()
        self.gradio = GradioConfig()
        self.finetune = FinetuneConfig()
        self.onnx = OnnxConfig()
        self.dataset = DatasetConfig()


config = Config()
