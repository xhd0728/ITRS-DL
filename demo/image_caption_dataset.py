import os
import json
import pandas as pd

from config import config
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor


def get_labels_and_cates(dataset, is_train):
    build_dataset_fn = {
        'mini imagenet': build_mini_imagenet_dataset,
        'coco': build_coco_dataset,
        'coco2017': build_coco2017_dataset,
        'flickr30k': build_flickr30k_dataset
    }
    array = build_dataset_fn[dataset](is_train=is_train)
    array = sorted(array, key=lambda x: int(x[1]))
    cates = [item[1] for item in array]

    labels = [item[3] for item in array]

    if isinstance(labels, list) and all(isinstance(sub_list, list) for sub_list in labels):
        len_list = list(map(lambda x: len(x), labels))
        labels = [item for sub_list in labels for item in sub_list]
        cates = [item for item, count in zip(
            cates, len_list) for _ in range(count)]
    return labels, cates


def build_mini_imagenet_dataset(is_train=False):
    root_dir = config.dataset.mini_imagenet_path

    with open(os.path.join(root_dir, 'classes_name.json'), 'r') as f:
        mini_imagenet_label = json.load(f)

    csv_filepath = 'new_train.csv' if is_train else 'new_val.csv'

    data = pd.read_csv(os.path.join(root_dir, csv_filepath))

    res = []
    for idx, (_, row) in enumerate(data.iterrows()):
        img_path = os.path.join(root_dir, 'images', row['filename'])
        category = int(mini_imagenet_label[row['label']][0])
        label = mini_imagenet_label[row['label']][1].replace('_', ' ')
        res.append((idx, category, img_path, label))
    return res


def build_coco_dataset(is_train=False):

    dataDir = config.dataset.coco_path
    dataType = 'train2014' if is_train else 'val2014'
    annFile = os.path.join(dataDir, f'annotations/captions_{dataType}.json')

    with open(annFile, 'r') as f:
        annotations = json.load(f)['annotations']

    imgid_to_captions = {}
    for ann in annotations:
        img_id = ann['image_id']
        caption = ann['caption']
        if img_id in imgid_to_captions:
            imgid_to_captions[img_id].append(caption)
        else:
            imgid_to_captions[img_id] = [caption]

    res = []
    img_folder = os.path.join(dataDir, dataType)
    for idx, (img_id, captions) in enumerate(imgid_to_captions.items()):
        img_info = {
            'id': img_id,
            'file_name': f'COCO_{dataType}_{img_id:012d}.jpg'
        }
        img_path = os.path.join(img_folder, img_info['file_name'])

        category = img_id
        res.append((idx, category, img_path, captions))
    return res


def build_coco2017_dataset(is_train=False):

    dataDir = config.dataset.coco2017_path
    dataType = 'train2017' if is_train else 'val2017'
    annFile = os.path.join(dataDir, f'annotations/captions_{dataType}.json')

    with open(annFile, 'r') as f:
        annotations = json.load(f)['annotations']

    imgid_to_captions = {}
    for ann in annotations:
        img_id = ann['image_id']
        caption = ann['caption']
        if img_id in imgid_to_captions:
            imgid_to_captions[img_id].append(caption)
        else:
            imgid_to_captions[img_id] = [caption]

    res = []
    img_folder = os.path.join(dataDir, dataType)
    for idx, (img_id, captions) in enumerate(imgid_to_captions.items()):
        img_info = {
            'id': img_id,
            'file_name': f'{img_id:012d}.jpg'
        }
        img_path = os.path.join(img_folder, img_info['file_name'])

        category = img_id
        res.append((idx, category, img_path, captions))
    return res


def build_flickr30k_dataset(is_train=False):
    dataDir = config.dataset.flickr30k_path
    annFile = os.path.join(dataDir, 'results_20130124.token')
    img_folder = os.path.join(dataDir, 'flickr30k-images')
    annotations = pd.read_table(
        annFile, sep='\t', header=None, names=['image', 'caption'])

    img_to_captions = {}
    for image, caption in zip(annotations['image'], annotations['caption']):
        image = str(image).split('#')[0]
        if image not in img_to_captions:
            img_to_captions[image] = [caption]
        else:
            img_to_captions[image].append(caption)

    res = []
    for idx, (image, captions) in enumerate(img_to_captions.items()):
        img_path = os.path.join(img_folder, image)
        category = int(image.split('.')[0])
        res.append((idx, category, img_path, captions))
    return res


class ImageCaptionDataset(Dataset):
    def __init__(self, is_train=False, return_loss=False, dataset='mini imagenet'):
        build_dataset_fn = {
            'mini imagenet': build_mini_imagenet_dataset,
            'coco': build_coco_dataset,
            'coco2017': build_coco2017_dataset,
            'flickr30k': build_flickr30k_dataset
        }
        self.data = build_dataset_fn[dataset](is_train=is_train)
        self.return_loss = return_loss
        self.processor = AutoProcessor.from_pretrained(
            config.finetune.checkpoint_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx, category, img_path, label = self.data[idx]
        img = Image.open(img_path)
        return idx, category, img, label

    def collate_fn(self, batch):
        ids, categories, images, labels = tuple(
            zip(*batch))

        if isinstance(labels, tuple) and all(isinstance(sub_list, list) for sub_list in labels):

            len_list = [len(x) for x in labels]

            flat_labels = [item for sub_list in labels for item in sub_list]

            output = self.processor(
                text=flat_labels, images=images, return_tensors='pt', padding=True)
            output['len_list'] = len_list
        else:
            output = self.processor(
                text=labels, images=images, return_tensors='pt', padding=True)

        if self.return_loss:
            output['return_loss'] = True
        return ids, categories, output
