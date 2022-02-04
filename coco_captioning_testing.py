from cgitb import small
from operator import gt
import os
from statistics import median
import this
from scipy import rand
import clip
import torch
#from torchvision.datasets import CIFAR100
import json
from PIL import Image
torch.manual_seed(0)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# prepare the image embedding
import mmcv
import math
import numpy as np

#json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json'

# load the json file
json_val = json.load(open(json_file_path))

# aggregate the annotation for each image
#file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
file_root = '/data2/lwll/zhuoming/detection/coco/train2017/'

from_img_id_to_bbox = {}
#{image_id:{image_name:"", bbox_list:[]},}
# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id != 472054 and image_id != 174228:
        continue
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'img_shape': (anno['width'], anno['height']), 'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    if image_id != 472054 and image_id != 174228:
        continue
    box = anno['bbox']
    box.append(anno['category_id'])
    from_img_id_to_bbox[image_id]['bbox'].append(box)

all_feature_res = {}
all_assigned_result = {}
#cate_result = []
file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)

# generate random bbox for the feat extraction
random_bbox = torch.randn(20,4)
random_bbox = torch.abs(random_bbox)
max_pos = torch.max(random_bbox)
random_bbox = random_bbox / max_pos
random_bbox = random_bbox.sort()[0]

torch.save(random_bbox, 'random_bbox_ratio.pt')


from_img_id_to_bbox[174228]['bbox'].append([278.87, 126.33, 150, 216.99, 1])
from_img_id_to_bbox[174228]['bbox'].append([278.87, 126.33, 170, 216.99, 1])
from_img_id_to_bbox[174228]['bbox'].append([278.87, 126.33, 190, 216.99, 1])
from_img_id_to_bbox[174228]['bbox'].append([278.87, 126.33, 210, 216.99, 1])
from_img_id_to_bbox[472054]['bbox'].append([302.25, 90.13, 130.6, 326.66, 1])
from_img_id_to_bbox[472054]['bbox'].append([282.25, 90.13, 150.6, 326.66, 1])
from_img_id_to_bbox[472054]['bbox'].append([262.25, 90.13, 170.6, 326.66, 1])
from_img_id_to_bbox[472054]['bbox'].append([242.25, 90.13, 190.6, 326.66, 1])
from_img_id_to_bbox[472054]['bbox'].append([222.25, 90.13, 210.6, 326.66, 1])

# go through all the image in the dict:
for count_i, image_id in enumerate(from_img_id_to_bbox.keys()):
    filenname = from_img_id_to_bbox[image_id]['path']
    img_w, img_h = from_img_id_to_bbox[image_id]['img_shape']
    #load the image and convert to numpy
    img_bytes = file_client.get(filenname)
    img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
    # the shape of the img shoud be (x, y, 3)
    image_result = []
    cate_result = []
    # obtain the gt bbox feat
    if len(from_img_id_to_bbox[image_id]['bbox']) != 0:
        for bbox in from_img_id_to_bbox[image_id]['bbox']:
            # for each bbox we need to calculate whether the bbox is inside the grid
            x, y, w, h, cat_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            #x_start_pos = math.floor(x)
            #y_start_pos = math.floor(y)
            #x_end_pos = math.ceil(x+w)
            #y_end_pos = math.ceil(y+h)
            # enlarge the image with 10% for each cord            
            #x_start_pos = math.floor(max(x-0.1*w, 0))
            #y_start_pos = math.floor(max(y-0.1*h, 0))
            #x_end_pos = math.ceil(min(x+1.1*w, img_w-1))
            #y_end_pos = math.ceil(min(y+1.1*h, img_h-1))
            x_start_pos = math.floor(max(x-0.25*w, 0))
            y_start_pos = math.floor(max(y-0.25*h, 0))
            x_end_pos = math.ceil(min(x+1.25*w, img_w-1))
            y_end_pos = math.ceil(min(y+1.25*h, img_h-1))
            # crop the square with respect to the gt bbox
            #if w > h:
            #    pad_size = (w - h) / 2
            #    x_start_pos = math.floor(x)
            #    y_start_pos = math.floor(max(y-pad_size, 0))
            #    x_end_pos = math.ceil(x+w)
            #    y_end_pos = math.ceil(min(y+h+pad_size, img_h-1))   
            #elif w < h:
            #    pad_size = (h - w) / 2
            #    x_start_pos = math.floor(max(x-pad_size, 0))
            #    y_start_pos = math.floor(y)
            #    x_end_pos = math.ceil(min(x+w+pad_size, img_w-1))
            #    y_end_pos = math.ceil(y+h)   
            #else:
            #    x_start_pos = math.floor(x)
            #    y_start_pos = math.floor(y)
            #    x_end_pos = math.ceil(x+w)
            #    y_end_pos = math.ceil(y+h)
            now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]           
            # crop the GT bbox and place it in the center of the zero square
            gt_h, gt_w, c = now_patch.shape
            if gt_h != gt_w:
                long_edge = max((gt_h, gt_w))
                empty_patch = np.zeros((long_edge, long_edge, 3))
                if gt_h > gt_w:
                    x_start = (long_edge - gt_w) // 2
                    x_end = x_start + gt_w
                    empty_patch[:, x_start: x_end] = now_patch
                else:
                    y_start = (long_edge - gt_h) // 2
                    y_end = y_start + gt_h
                    empty_patch[y_start: y_end] = now_patch
                now_patch = empty_patch
            #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
            # convert the numpy to PIL image
            PIL_image = Image.fromarray(np.uint8(now_patch))
            # do the preprocessing
            new_patch = preprocess(PIL_image)
            #image_result.append(np.expand_dims(new_patch, axis=0))
            image_result.append(new_patch.unsqueeze(dim=0))
            cate_result.append(cat_id)
    # obtain random bbox feat
    for bbox_ratio in random_bbox:
        x_start_pos = math.floor(bbox_ratio[0] * img_w)
        y_start_pos = math.floor(bbox_ratio[1] * img_h)
        x_end_pos = math.ceil(bbox_ratio[2] * img_w)
        y_end_pos = math.ceil(bbox_ratio[3] * img_h)
        now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
        # convert the numpy to PIL image
        PIL_image = Image.fromarray(np.uint8(now_patch))
        # do the preprocessing
        new_patch = preprocess(PIL_image)
        #image_result.append(np.expand_dims(new_patch, axis=0))
        image_result.append(new_patch.unsqueeze(dim=0))
    cropped_patches = torch.cat(image_result, dim=0).cuda()
    with torch.no_grad():
        image_features = model.encode_image(cropped_patches)
    #print(image_features.shape)
    all_feature_res[image_id] = image_features.cpu()
    all_assigned_result[image_id] = torch.tensor(cate_result)
    #if count_i > 100:
    #    break
    if count_i % 1000 == 0:
        print(count_i)


torch.save(all_feature_res, '174228_and_472054_feat.pt')
torch.save(all_assigned_result, '174228_and_472054_assigned_result.pt')


text_embedding = torch.load('coco_name_feat_combined.pt')
person_with_dog = torch.load('person_with_dog_feat.pt').cuda()
person_with_dog_sentence = torch.load('person_with_dog_sentence_feat.pt').cuda()
person_with_kite = torch.load('person_with_kite_feat.pt').cuda()
person_with_kite_sentence = torch.load('person_with_kite_sentence_feat.pt').cuda()
text_embedding = torch.cat([text_embedding, person_with_dog, person_with_dog_sentence, person_with_kite, person_with_kite_sentence])
#text_embedding = torch.load('coco_name_feat_multi_template.pt')
#text_embedding = torch.load('coco_name_feat_8_template.pt')
#text_embedding = torch.load('coco_name_feat_cifar10_template.pt')
#image_embedding = torch.load('val_img_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_enlarged_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_center_pad_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_zero_pad_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_1_2times_zero_pad_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_1_5times_zero_pad_gt_rand_feat_res.pt')
image_embedding = torch.load('174228_and_472054_feat.pt')

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

gt_predict_result = {}
rand_predict_result = {}

all_gt_entropy = []
all_rand_entropy = []

total_gt_entry = 0
total_rand_entry = 0

for key in image_embedding.keys():
    #if key != 174228:
    #    continue
    image_features = image_embedding[key].cuda()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_embedding.T).softmax(dim=-1)
    # calculate the entropy
    gt_pred = similarity[:-20]
    rand_pred = similarity[-20:]
    gt_predict_result[key] = gt_pred
    rand_predict_result[key] = rand_pred
    # calculate the gt entropy
    if gt_pred.shape[0] != 0:
        # deal with the 0 in the gt_pred
        gt_pred[gt_pred == 0] = 1e-5
        log_result = - torch.log(gt_pred)
        gt_entro = (log_result * gt_pred).sum(dim=-1)
        all_gt_entropy.append(gt_entro)
        #print(gt_entro.shape)
        #gt_entropy = (log_result * gt_pred).sum() 
        #total_gt_entry += gt_pred.shape[0]
        #all_gt_entropy.append(gt_entropy.item())
    # calculate the random entropy
    # deal with the 0 in the rand_pred
    rand_pred[rand_pred == 0] = 1e-5
    log_result = - torch.log(rand_pred)
    rand_entro = (log_result * rand_pred).sum(dim=-1)
    all_rand_entropy.append(rand_entro)
    #print(rand_entro.shape)
    #rand_entropy = (log_result * rand_pred).sum()
    #total_rand_entry += rand_pred.shape[0]
    #all_rand_entropy.append(rand_entropy.item())


coco_cate = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'person_with_dog', 'person_with_dog_sentence', 'person_with_kite', 'person_with_kite_sentence']


testing = gt_predict_result[174228]
#testing = gt_predict_result[554002]
#testing = random_predict_result[507037]
#testing = random_predict_result[580410]
#testing = random_predict_result[275198]

for i, similarity in enumerate(testing):
    values, indices = similarity.topk(5)
    # Print the result
    print("Image " + str(i) +" Top predictions:\n")
    for value, index in zip(values, indices):
        #print(imagenet_cate_name[str(index.item())], 100 * value.item())
        print(coco_cate[index.item()], 100 * value.item())
        #print(f"{imagenet_cate_name[str(index.item())]:>16s}: {100 * value.item():.2f}%")


testing = gt_predict_result[472054]
#testing = gt_predict_result[554002]
#testing = random_predict_result[507037]
#testing = random_predict_result[580410]
#testing = random_predict_result[275198]

for i, similarity in enumerate(testing):
    values, indices = similarity.topk(5)
    # Print the result
    print("Image " + str(i) +" Top predictions:\n")
    for value, index in zip(values, indices):
        #print(imagenet_cate_name[str(index.item())], 100 * value.item())
        print(coco_cate[index.item()], 100 * value.item())
        #print(f"{imagenet_cate_name[str(index.item())]:>16s}: {100 * value.item():.2f}%")

