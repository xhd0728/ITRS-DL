import json
import os
import pandas as pd
from tqdm import tqdm
from config import config
from log_handler import Logger

logger = Logger.get_logger()


def process_json(json_path, image_folder, output_csv):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    result_df = pd.DataFrame(columns=['id', 'filename'])

    for annotation in tqdm(data['annotations']):
        image_id = annotation['image_id']

        filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = os.path.join(image_folder, filename)

        if os.path.exists(image_path):
            result_df = pd.concat([result_df, pd.DataFrame(
                {'id': [image_id], 'filename': [filename]})], ignore_index=True)

    result_df = result_df.drop_duplicates()

    result_df.to_csv(output_csv, index=False)
    logger.info(f"saved at {output_csv}")


if __name__ == '__main__':

    json_path = os.path.join(
        config.dataset.coco_path,
        'annotations/captions_train2014.json'
    )
    image_folder = os.path.join(
        config.dataset.coco_path,
        'train2014'
    )
    output_csv = os.path.join(
        config.dataset.coco_path,
        'coco2014_train.csv'
    )

    process_json(json_path, image_folder, output_csv)
