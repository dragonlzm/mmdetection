from numpy.lib.type_check import imag
from torch import tensor
import mmcv
import numpy as np
import json
import math
import os
import clip
import torch
import io
from PIL import Image

# prepare the model
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

file_name = '/data2/lwll/zhuoming/detection/coco/val2017/000000281693.jpg'
file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)
img_bytes = file_client.get(file_name)
img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
PIL_image = Image.fromarray(np.uint8(img))
# do the preprocessing
new_patch = preprocess(PIL_image)

# save the image after processing
#np_result = new_patch.permute(1, 2, 0).numpy()
#data = Image.fromarray(np.uint8(np_result))
#data.save('000000281693_cropped.jpg')

with torch.no_grad():
    image_features = model.encode_image(new_patch.unsqueeze(dim=0).to(device))