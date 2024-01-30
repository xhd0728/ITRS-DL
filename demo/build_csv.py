import json
import os
import pandas as pd
from tqdm import tqdm
from config import config
from log_handler import Logger

logger = Logger.get_logger()


def coco_process_json(json_path, image_folder, output_csv):
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


def flickr_process_token(token_path, image_folder, output_csv):
    with open(token_path, 'r') as token_file:
        lines = token_file.readlines()

    result_df = pd.DataFrame(columns=['id', 'filename'])

    for line in tqdm(lines):

        filename = line.split('#')[0]
        image_path = os.path.join(image_folder, filename)

        if os.path.exists(image_path):
            result_df = pd.concat([result_df, pd.DataFrame(
                {'id': [filename.split('.')[0]], 'filename': [filename]})], ignore_index=True)
    result_df = result_df.drop_duplicates()

    result_df.to_csv(output_csv, index=False)
    logger.info(f"saved at {output_csv}")


if __name__ == '__main__':

    # json_path = os.path.join(
    #     config.dataset.coco_path,
    #     'annotations/captions_train2014.json'
    # )
    token_path = os.path.join(
        config.dataset.flickr30k_path,
        'results_20130124.token'
    )
    # image_folder = os.path.join(
    #     config.dataset.coco_path,
    #     'train2014'
    # )
    image_folder = os.path.join(
        config.dataset.flickr30k_path,
        'flickr30k-images'
    )
    # output_csv = os.path.join(
    #     config.dataset.coco_path,
    #     'coco2014_train.csv'
    # )
    output_csv = os.path.join(
        config.dataset.flickr30k_path,
        'flickr30k.csv'
    )

    # coco_process_json(json_path, image_folder, output_csv)
    flickr_process_token(token_path, image_folder, output_csv)
