# usage for webdemo

## 1. Install required dependencies

```shell
pip install -r requirements.txt
```

notice, the better way is install the pytorch first, then delete the torch==2.0.1+cu118 from the requirements.txt, and run the script above

## 2. prepare the dataset and clip model

- coco2014/2017: https://cocodataset.org/#download

```
coco2014/2017
├── annotations
│   ├── captions_train2014.json
│   └── captions_val2014.json
├── train2014/2017
│   ├── xxx.jpg
│   └── ...
└── val2014/2017
    ├── xxx.jpg
    └── ...
```

- mini imagenet: https://pan.baidu.com/share/init?surl=Uro6RuEbRGGCQ8iXvF2SAQ&pwd=hl31

```
mini-imagenet
├── classes_name.json
├── imagenet_class_index.json
├── images
│   ├── xxx.jpg
│   ├── xxx.jpg
│   └── ...
├── new_train.csv
├── new_val.csv
├── train.json
└── val.json
```

- flickr30k: http://shannon.cs.illinois.edu/DenotationGraph/data/index.html

```
flickr30k
├── flickr30k-images
│   ├── xxx.jpg
│   └── ...
└── results_20130124.token
```

- clip model

    - openai/clip-vit-large-patch14 (recommend)
    - openai/clip-vit-base-patch32 (recommend)
    - openai/clip-vit-base-patch16

## 3. set the configure in config.py

## 4. bin -> pt -> onnx

```bash
python export_onnx.py
```

## 5. run web demo

```bash
python web_demo.py
```