# Desc: 配置文件
model_name = 'clip-vit-large-patch14'


class MilvusConfig:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = '19530'
        self.collection_name = 'large_patch14_train_coco2014'
        self.vector_dim = 768
        self.topk = 10

        self.index_params = {
            'metric_type': 'IP',  # 内积距离
            'index_type': 'HNSW',  # 算法类型
            'params': {
                'M': 8,
                'efConstruction': 128
            }
        }

        self.search_params = {
            "metric_type": 'IP',
            "params": {
                "ef": 20
            }  # topk < search_param的ef
        }


class RedisConfig:
    def __init__(self):
        self.use_redis = False
        self.host = '127.0.0.1'
        self.port = 6379
        self.db = 0
        self.expire_time = 3600  # 过期时间


class GradioConfig:
    def __init__(self):
        # 模型的保存路径
        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'


class FinetuneConfig:
    def __init__(self):
        # 模型的保存路径
        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'
        self.save_dir = './checkpoint'


class OnnxConfig:
    def __init__(self):
        self.use_onnx = True
        # 模型的保存路径
        self.checkpoint_dir = f'/mnt/f/DLWorks/model/{model_name}'
        self.save_dir = './onnx'


class DatasetConfig:
    def __init__(self):
        # 数据集选择
        # coco2014->coco
        # coco2017->coco2017
        # flickr30k->flickr30k
        # mini imagenet->mini imagenet
        self.name = 'coco'

        # 数据集路径
        self.mini_imagenet_path = '/mnt/f/DLWorks/dataset/mini-imagenet'
        self.flickr30k_path = '/mnt/f/DLWorks/dataset/flickr30k'
        self.coco_path = '/mnt/f/DLWorks/dataset/mscoco2014'
        self.coco2017_path = '/mnt/f/DLWorks/dataset/mscoco2017'

        # 数据集划分
        self.is_train = True


class Config:
    def __init__(self):
        self.milvus = MilvusConfig()
        self.redis = RedisConfig()
        self.gradio = GradioConfig()
        self.finetune = FinetuneConfig()
        self.onnx = OnnxConfig()
        self.dataset = DatasetConfig()


config = Config()
