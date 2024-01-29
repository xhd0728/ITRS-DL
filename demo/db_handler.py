import pickle
import redis
from config import config
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from log_handler import Logger

logger = Logger.get_logger()


class MilvusHandler:
    def __init__(
        self,
        host=config.milvus.host,
        port=config.milvus.port,
        collection_name=config.milvus.collection_name
    ):

        self.collection_name = collection_name
        connections.connect(host=host, port=port)

        self._connect_collection(self.collection_name)

    def _connect_collection(self, collection_name):
        if utility.has_collection(collection_name):
            self.collection = Collection(collection_name)
            self.collection.load()
        else:
            logger.info(
                f'there is no collection corresponding to {collection_name} in milvus')

    def search(self, embeds, topk):
        res = self.collection.search(
            data=embeds,
            anns_field='embedding',
            param=config.milvus.search_params,
            limit=topk,
            output_fields=['category']
        )

        ids = [list(hits.ids) for hits in res]
        distances = [list(hits.distances) for hits in res]
        categories = [[hit.entity.get('category')
                       for hit in hits] for hits in res]
        return ids, distances, categories

    def insert(self, data):
        return self.collection.insert(data=data)

    def load_and_flush(self):
        self.collection.load()
        self.collection.flush()

    def get_num_entities(self):
        logger.info(
            f'{self.collection.num_entities} pieces of data have been inserted')

    @staticmethod
    def create_collection(collection_name, dim):
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

        collection.create_index(field_name="embedding",
                                index_params=index_params)
        return collection

    @staticmethod
    def list_collections():
        logger.info(utility.list_collections())

    @staticmethod
    def drop_collection(collection_name):
        utility.drop_collection(collection_name)
        logger.info(f'deleted {collection_name}')


class RedisHandler:
    def __init__(
        self,
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db
    ):

        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def set_serialized(self, key, value, ex=config.redis.expire_time):
        serialized_value = pickle.dumps(value)
        self.redis_client.set(key, serialized_value, ex)

    def set(self, key, value, ex=config.redis.expire_time):
        self.redis_client.set(key, value, ex)

    def get(self, key):
        result = self.redis_client.get(key)
        if result:
            return pickle.loads(result)
        return None

    def mget(self, keys):
        deserialize_res = []
        results = self.redis_client.mget(keys)
        for result in results:
            if result:
                deserialize_res.append(pickle.loads(result))
            else:
                deserialize_res.append(None)
        return deserialize_res

    def update_serialized(self, key, new_value):
        serialized_value = pickle.dumps(new_value)
        self.redis_client.set(key, serialized_value)

    def update_data(self, key, new_value):
        self.redis_client.set(key, new_value)

    def delete_data(self, key):
        self.redis_client.delete(key)
