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
#model, preprocess = clip.load('RN50x16', device)

json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
# load the json file
json_val = json.load(open(json_file_path))

# aggregate the annotation for each image
file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
from_img_id_to_bbox = {}
#{image_id:{image_name:"", bbox_list:[]},}
# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    box = anno['bbox']
    box.append(anno['category_id'])
    from_img_id_to_bbox[image_id]['bbox'].append(box)

all_feature_res = []

# go through all the image in the dict:
cate_result = []
file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)


for count_i, image_id in enumerate(from_img_id_to_bbox.keys()):
    filenname = from_img_id_to_bbox[image_id]['path']

    #load the image and convert to numpy
    #filenname = '/data2/lwll/zhuoming/detection/coco/val2017/000000581100.jpg'
    img_bytes = file_client.get(filenname)
    img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
    #img = mmcv.imfrombytes(img_bytes, flag='color')

    # the shape of the img shoud be (x, y, 3)
    image_result = []

    if len(from_img_id_to_bbox[image_id]['bbox']) == 0:
        continue

    for bbox in from_img_id_to_bbox[image_id]['bbox']:
        # for each bbox we need to calculate whether the bbox is inside the grid
        x, y, w, h, cat_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]

        x_start_pos = math.floor(x)
        y_start_pos = math.floor(y)
        x_end_pos = math.ceil(x+w)
        y_end_pos = math.ceil(y+h)

        now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
        #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
        # convert the numpy to PIL image
        PIL_image = Image.fromarray(np.uint8(now_patch))
        # do the preprocessing
        new_patch = preprocess(PIL_image)
        #print(new_patch.shape)
        #break
        #image_result.append(np.expand_dims(new_patch, axis=0))
        image_result.append(new_patch.unsqueeze(dim=0))

        cate_result.append(cat_id)

    #cropped_patches = np.concatenate(image_result, axis=0)
    cropped_patches = torch.cat(image_result, dim=0).cuda()
    #print(cropped_patches.shape)
    #cropped_patches_tensor = torch.from_numpy(cropped_patches)
    #cropped_patches_tensor = torch.permute(cropped_patches_tensor, (0, 3, 1, 2)).cuda()
    with torch.no_grad():
        image_features = model.encode_image(cropped_patches)
    #print(image_features.shape)
    all_feature_res.append(image_features.cpu())
#del image_features

# Save to file
all_assigned_res = torch.tensor(cate_result)
all_feature_res = torch.cat(all_feature_res, dim=0)
np.save('assigned_gt_res.npy', all_assigned_res.reshape(-1).numpy())
#torch.save(all_assigned_res, 'assigned_gt_res.pt')
torch.save(all_feature_res, 'gt_feature.pt')



