import os
import onnxruntime
import torch
import numpy as np

from config import config
from transformers import ChineseCLIPProcessor, ChineseCLIPTextModel

from log_handler import Logger

logger = Logger.get_logger()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = ChineseCLIPProcessor.from_pretrained(config.onnx.checkpoint_dir)
text_encoder_hf = ChineseCLIPTextModel.from_pretrained(
    config.onnx.checkpoint_dir,
    ignore_mismatched_sizes=True
).to(device)
model_name = config.onnx.checkpoint_dir.split('/')[-1]


os.makedirs('./onnx', exist_ok=True)
torch.save(text_encoder_hf, f'./onnx/{model_name}_text_encoder.pt')
text_encoder_pt = torch.load(f'./onnx/{model_name}_text_encoder.pt')


logger.info(f'bin->pt completed')


text = processor(
    text=['hello' for _ in range(32)],
    return_tensors='pt',
    max_length=77,
    padding=True
).to(device)
text = dict(text)
torch.onnx.export(
    model=text_encoder_pt,
    args=text,
    f=f'./onnx/{model_name}_text_encoder.onnx',
    opset_version=14,
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['text_embeds', 'last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'input_ids_dim'},
        'attention_mask': {0: 'batch_size', 1: 'attention_mask_dim'},
        'text_embeds': {0: 'batch_size'},
        'last_hidden_state': {0: 'batch_size'}
    }
)

logger.info(f'pt->onnx completed')


out = text_encoder_hf(**text)


providers = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'
session = onnxruntime.InferenceSession(
    f'./onnx/{model_name}_text_encoder.onnx', providers=[providers])
onnx_text = processor(text='hello', return_tensors='np',
                      max_length=77, padding=True)
onnx_text = dict(onnx_text)
for i in onnx_text:
    onnx_text[i] = onnx_text[i].astype(np.int64)
out = session.run(None, onnx_text)[0]
