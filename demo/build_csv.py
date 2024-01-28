import json
import os
import pandas as pd
from tqdm import tqdm
from config import config


def process_json(json_path, image_folder, output_csv):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    result_df = pd.DataFrame(columns=['id', 'filename'])

    # 遍历annotations
    for annotation in tqdm(data['annotations']):
        image_id = annotation['image_id']

        # 根据数据集的格式做相应的修改
        # 这里以COCO2014为例
        filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = os.path.join(image_folder, filename)

        if os.path.exists(image_path):
            result_df = pd.concat([result_df, pd.DataFrame(
                {'id': [image_id], 'filename': [filename]})], ignore_index=True)

    # 去除重复项
    result_df = result_df.drop_duplicates()
    # 保存结果到CSV
    result_df.to_csv(output_csv, index=False)
    print(f"结果已保存在 {output_csv}")


# 替换为你的JSON文件路径、图片文件夹路径和输出CSV路径
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

# 执行程序
process_json(json_path, image_folder, output_csv)
