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

# prepare the imagenet cate text embeddings
'''
# load the word embedding of the imagenet
#imagenet_cate_name = json.load(open("C:\\Users\\XPS\\Desktop\\imagenet1000_clsidx_to_labels.json"))
imagenet_cate_name = json.load(open('/data2/lwll/zhuoming/code/CLIP/imagenet1000_converted_name.json'))

all_cate_feat = []
for cate_id in imagenet_cate_name:
    result_for_cate = []
    for name in imagenet_cate_name[cate_id]:
        tokenized_result = clip.tokenize(f"a photo of a {name}")
        result_for_cate.append(tokenized_result)
    #single_tokenized = result_for_cate[0].to(device)
    cate_tokenized = torch.cat(result_for_cate).to(device)
    with torch.no_grad():
        cate_features = model.encode_text(cate_tokenized)
    single_feat = cate_features[0]
    cate_features = cate_features.mean(dim=0)
    #print(torch.norm(cate_features-single_feat))
    all_cate_feat.append(cate_features.unsqueeze(dim=0))

all_cate_feat = torch.cat(all_cate_feat, dim=0)
torch.save(all_cate_feat, 'imagenet_name_feat_combined.pt')'''


# prepare the coco cate text embeddings
#sentence_template = ['There is {category} in the scene.',
#'There is the {category} in the scene.',
#'a photo of {category} in the scene.',
#'a photo of the {category} in the scene.',
#'a photo of one {category} in the scene.',
#'a photo of {category}.',
#'a photo of my {category}.',
#'a photo of the {category}.',
#'a photo of one {category}.',]
#'a photo of many {category}.',
#'a good photo of {category}.',
#'a good photo of the {category}.',
#'a bad photo of {category}.',
#'a bad photo of the {category}.',
#'a photo of a nice {category}.',
#'a photo of the nice {category}.',
#'a photo of a cool {category}.',
#'a photo of the cool {category}.',
#'a photo of a weird {category}.',
#'a photo of the weird {category}.',
#'a photo of a small {category}.',
#'a photo of the small {category}.',
#'a photo of a large {category}.',
#'a photo of the large {category}.',
#'a photo of a clean {category}.',
#'a photo of the clean {category}.',
#'a photo of a dirty {category}.',
#'a photo of the dirty {category}.']

sentence_template = ['a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',]


json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'

coco_json = json.load(open(json_file_path))

all_cate_feat = []
for cate_info in coco_json['categories']:
    name = cate_info['name']
    # handle the space in the cate_name
    #if ' ' in name:
    #    name = name.replace(' ', '_')
    all_sentences_result = []
    for template in sentence_template:
        #now_sentence = template.replace('{category}', name)
        now_sentence = template.replace('{}', name)
        print(now_sentence)
        #tokenized_result = clip.tokenize(f"a photo of a {name}").to(device)
        tokenized_result = clip.tokenize(now_sentence).to(device)
        all_sentences_result.append(tokenized_result)
    all_sentences_result = torch.cat(all_sentences_result, dim=0)

    #tokenized_result = clip.tokenize(f"a photo of a {name}").to(device)    
    #print(all_sentences_result.shape)
    with torch.no_grad():
        #cate_features = model.encode_text(tokenized_result)
        cate_features = model.encode_text(all_sentences_result)
    #print(cate_features.shape)
    cate_features = cate_features.mean(dim=0, keepdim=True)
    #print(cate_features.shape)
    #print(torch.norm(cate_features-single_feat))
    all_cate_feat.append(cate_features)

all_cate_feat = torch.cat(all_cate_feat, dim=0)
print(all_cate_feat.shape)

#torch.save(all_cate_feat, 'coco_name_feat_combined.pt')
#torch.save(all_cate_feat, 'coco_name_feat_space_replaced.pt')

#torch.save(all_cate_feat, 'coco_name_feat_multi_template.pt')
torch.save(all_cate_feat, 'coco_name_feat_8_template.pt')


# prepare the image embedding
import mmcv
import math
import numpy as np

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
        from_img_id_to_bbox[image_id] = {'img_shape': (anno['width'], anno['height']), 'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
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

#torch.save(all_feature_res, '100img_gt_rand_feat_res.pt')
#torch.save(all_feature_res, 'val_img_gt_rand_feat_res.pt')
#torch.save(all_assigned_result, 'val_img_gt_rand_all_assigned_result.pt')
#torch.save(all_feature_res, 'val_img_enlarged_gt_rand_feat_res.pt')
#torch.save(all_assigned_result, 'val_img_enlarged_gt_rand_all_assigned_result.pt')
#torch.save(all_feature_res, 'val_img_center_pad_gt_rand_feat_res.pt')
#torch.save(all_assigned_result, 'val_img_center_pad_gt_rand_all_assigned_result.pt')
#torch.save(all_feature_res, 'val_img_zero_pad_gt_rand_feat_res.pt')
#torch.save(all_assigned_result, 'val_img_zero_pad_gt_rand_all_assigned_result.pt')
#torch.save(all_feature_res, 'val_img_1_2times_zero_pad_gt_rand_feat_res.pt')
#torch.save(all_assigned_result, 'val_img_1_2times_zero_pad_gt_rand_all_assigned_result.pt')
torch.save(all_feature_res, 'val_img_1_5times_zero_pad_gt_rand_feat_res.pt')
torch.save(all_assigned_result, 'val_img_1_5times_zero_pad_gt_rand_all_assigned_result.pt')

# predict the categories (calculate the entropy)
#text_embedding = torch.load('imagenet_name_feat_combined.pt')
#image_embedding = torch.load('100img_gt_rand_feat_res.pt')

#text_embedding = torch.load('coco_name_feat_space_replaced.pt')
text_embedding = torch.load('coco_name_feat_combined.pt')
#text_embedding = torch.load('coco_name_feat_multi_template.pt')
#text_embedding = torch.load('coco_name_feat_8_template.pt')
#text_embedding = torch.load('coco_name_feat_cifar10_template.pt')
#image_embedding = torch.load('val_img_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_enlarged_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_center_pad_gt_rand_feat_res.pt')
image_embedding = torch.load('val_img_zero_pad_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_1_2times_zero_pad_gt_rand_feat_res.pt')
#image_embedding = torch.load('val_img_1_5times_zero_pad_gt_rand_feat_res.pt')

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

gt_predict_result = {}
rand_predict_result = {}

all_gt_entropy = []
all_rand_entropy = []

total_gt_entry = 0
total_rand_entry = 0

for key in image_embedding.keys():
    #if key != 275198:
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

all_gt_entropy = torch.cat(all_gt_entropy)
print(all_gt_entropy.shape)
all_rand_entropy = torch.cat(all_rand_entropy)
print(all_rand_entropy.shape)

print('gt_mean', torch.mean(all_gt_entropy))
print('rand_mean', torch.mean(all_rand_entropy))
print('gt_std', torch.std(all_gt_entropy))
print('rand_std', torch.std(all_rand_entropy))

#final_gt_entro = sum(all_gt_entropy) / total_gt_entry
#final_rand_entro = sum(all_rand_entropy) / total_rand_entry
#print('final_gt_entro: ', final_gt_entro, 'final_rand_entro', final_rand_entro)

#torch.save(gt_predict_result, 'coco80_gt_predict_result.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_enlarged.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_enlarged_multi_template.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged_multi_template.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_enlarged_8_template.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged_8_template.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_enlarged_cifar10_template.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged_cifar10_template.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_center_padded.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_center_padded.pt')

torch.save(gt_predict_result, 'coco80_gt_predict_result_zero_padded.pt')
torch.save(rand_predict_result, 'coco80_rand_predict_result_zero_padded.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_1_2times_zero_padded.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_1_2times_zero_padded.pt')

#torch.save(gt_predict_result, 'coco80_gt_predict_result_1_5times_zero_padded.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result_1_5times_zero_padded.pt')

#torch.save(gt_predict_result, 'imagenet_gt_predict_result_enlarged.pt')
#torch.save(rand_predict_result, 'imagenet_rand_predict_result_enlarged.pt')

# Print the result
#print("\nTop predictions:\n")
#for value, index in zip(values, indices):
#    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")


'''
# for predict acc
gt_predict_result = torch.load('coco80_gt_predict_result.pt')
rand_predict_result = torch.load('coco80_rand_predict_result.pt')

# from coco cate_id to text embedding idx
json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
json_val = json.load(open(json_file_path))

# aggregate the annotation for each image
file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
from_img_id_to_bbox = {}
# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'img_shape': (anno['width'], anno['height']), 'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    box = anno['bbox']
    box.append(anno['category_id'])
    from_img_id_to_bbox[image_id]['bbox'].append(box)

from_cate_id_to_embedding_id = {}
for i, cate_anno in enumerate(json_val['categories']):
    cate_id = cate_anno['id']
    from_cate_id_to_embedding_id[cate_id] = i

all_acc = 0
all_gt_num = 0
for key in gt_predict_result.keys():
    #print(key)
    gt_pred = gt_predict_result[key]
    pred_idx = torch.argmax(gt_pred, dim=1)
    # convert gt label to the embedding_idx
    gt_gt_label = []
    for bbox in from_img_id_to_bbox[key]['bbox']:
        #print(bbox)
        cate_id = bbox[-1]
        #print(cate_id)
        gt_gt_label.append(from_cate_id_to_embedding_id[cate_id])
    gt_num = len(gt_gt_label)
    gt_gt_label = torch.tensor(gt_gt_label)
    acc = (gt_gt_label.cuda() == pred_idx).sum()
    all_acc += acc.item()
    all_gt_num += gt_num

all_acc /= all_gt_num
print(all_acc)    '''
    

# predict acc with assignment result

#gt_predict_result = torch.load('coco80_gt_predict_result.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_enlarged.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_enlarged_multi_template.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_enlarged_8_template.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_enlarged_cifar10_template.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_center_padded.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_zero_padded.pt')
gt_predict_result = torch.load('coco80_gt_predict_result_1_2times_zero_padded.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_1_5times_zero_padded.pt')

#gt_assignment_res = torch.load('val_img_enlarged_gt_rand_all_assigned_result.pt')
gt_assignment_res = torch.load('val_img_center_pad_gt_rand_all_assigned_result.pt')
#gt_assignment_res = torch.load('val_img_zero_pad_gt_rand_all_assigned_result.pt')


json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
json_val = json.load(open(json_file_path))

from_cate_id_to_embedding_id = {}
from_embedding_id_to_cate_name = {}
for i, cate_anno in enumerate(json_val['categories']):
    cate_id = cate_anno['id']
    from_cate_id_to_embedding_id[cate_id] = i
    from_embedding_id_to_cate_name[i] = cate_anno['name']

all_acc = 0
all_gt_num = 0
for key in gt_predict_result.keys():
    #print(key)
    gt_pred = gt_predict_result[key]
    assigned_cate_ids = gt_assignment_res[key]
    #print(assigned_cate_ids)
    pred_idx = torch.argmax(gt_pred, dim=1)
    # convert gt label to the embedding_idx
    gt_gt_label = [from_cate_id_to_embedding_id[cate_id.item()] for cate_id in assigned_cate_ids]
    gt_num = len(gt_gt_label)
    gt_gt_label = torch.tensor(gt_gt_label)
    acc = (gt_gt_label.cuda() == pred_idx).sum()
    all_acc += acc.item()
    all_gt_num += gt_num

all_acc /= all_gt_num
print(all_acc)

# obtain the confusion matrix
confusion_matrix = torch.zeros(80, 80)

for key in gt_predict_result.keys():
    #print(key)
    gt_pred = gt_predict_result[key]
    assigned_cate_ids = gt_assignment_res[key]
    #print(assigned_cate_ids)
    pred_idxes = torch.argmax(gt_pred, dim=1)
    # convert gt label to the embedding_idx
    for gt_cate_id, pred_idx in zip(assigned_cate_ids, pred_idxes):
        gt_idx = from_cate_id_to_embedding_id[gt_cate_id.item()]
        confusion_matrix[gt_idx][pred_idx.item()] += 1

print(confusion_matrix)

normalized_mat = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True)
print(normalized_mat)

import pandas as pd
import numpy as np

temp = normalized_mat.numpy()
temp = pd.DataFrame(temp)
#temp.to_csv('normalized_mat.csv')
temp.to_csv('normalized_mat_1_2time_zero_padding.csv')



# for the prediction top 5 prob
#imagenet_cate_name = json.load(open('/data2/lwll/zhuoming/code/CLIP/imagenet1000_converted_name.json'))
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
'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

#rand_predict_result = torch.load('imagenet_rand_predict_result_enlarged.pt')
#testing = rand_predict_result[507037]
#testing = rand_predict_result[580410]
#testing = rand_predict_result[275198]

#torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_enlarged.pt')
gt_predict_result = torch.load('coco80_gt_predict_result_zero_padded.pt')
#random_predict_result = torch.load('coco80_rand_predict_result_enlarged.pt')
#testing = gt_predict_result[507037]
#testing = gt_predict_result[580410]
#testing = gt_predict_result[275198]
testing = gt_predict_result[554002]


#testing = random_predict_result[507037]
#testing = random_predict_result[580410]
#testing = random_predict_result[275198]

for similarity in testing:
    values, indices = similarity.topk(5)
    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        #print(imagenet_cate_name[str(index.item())], 100 * value.item())
        print(coco_cate[index.item()], 100 * value.item())
        #print(f"{imagenet_cate_name[str(index.item())]:>16s}: {100 * value.item():.2f}%")


# small medium and large 
import mmcv
import math
import numpy as np

json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
# load the json file
json_val = json.load(open(json_file_path))

areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]

# aggregate the annotation for each image
file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
from_img_id_to_bbox = {}
#{image_id:{image_name:"", bbox_list:[]},}
# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'img_shape': (anno['width'], anno['height']), 'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    box = anno['bbox']
    box.append(anno['category_id'])
    from_img_id_to_bbox[image_id]['bbox'].append(box)

all_size_result = {}

# go through all the image in the dict:
for count_i, image_id in enumerate(from_img_id_to_bbox.keys()):
    # the shape of the img shoud be (x, y, 3)
    size_result = []
    # obtain the gt bbox feat
    if len(from_img_id_to_bbox[image_id]['bbox']) != 0:
        for bbox in from_img_id_to_bbox[image_id]['bbox']:
            # for each bbox we need to calculate whether the bbox is inside the grid
            x, y, w, h, cat_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
            area = w * h
            if area < 32 ** 2:
                tag = 0 
            elif area > 96 ** 2:
                tag = 2
            else:
                tag = 1
            size_result.append(tag)
    all_size_result[image_id] = torch.tensor(size_result)
    #if count_i > 100:
    #    break
    if count_i % 1000 == 0:
        print(count_i)

torch.save(all_size_result, 'val_img_size_tags.pt')



# acc with scale
gt_predict_result = torch.load('coco80_gt_predict_result_1_2times_zero_padded.pt')
#gt_predict_result = torch.load('coco80_gt_predict_result_1_5times_zero_padded.pt')

#gt_assignment_res = torch.load('val_img_enlarged_gt_rand_all_assigned_result.pt')
gt_assignment_res = torch.load('val_img_center_pad_gt_rand_all_assigned_result.pt')
#gt_assignment_res = torch.load('val_img_zero_pad_gt_rand_all_assigned_result.pt')
all_size_tags = torch.load('val_img_size_tags.pt')

json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
json_val = json.load(open(json_file_path))

from_cate_id_to_embedding_id = {}
from_embedding_id_to_cate_name = {}
for i, cate_anno in enumerate(json_val['categories']):
    cate_id = cate_anno['id']
    from_cate_id_to_embedding_id[cate_id] = i
    from_embedding_id_to_cate_name[i] = cate_anno['name']

all_acc = [0, 0, 0]
all_gt_num = [0, 0, 0]
for key in gt_predict_result.keys():
    #print(key)
    gt_pred = gt_predict_result[key]
    assigned_cate_ids = gt_assignment_res[key]
    this_size_tags = all_size_tags[key]
    small_idx = (this_size_tags == 0)
    median_idx = (this_size_tags == 1)
    large_idx = (this_size_tags == 2)
    pred_idx = torch.argmax(gt_pred, dim=1)
    # convert gt label to the embedding_idx
    gt_gt_label = [from_cate_id_to_embedding_id[cate_id.item()] for cate_id in assigned_cate_ids]
    gt_gt_label = torch.tensor(gt_gt_label)
    # all predict result
    match_res = (gt_gt_label.cuda() == pred_idx)
    small_match_res = match_res[small_idx]
    median_match_res = match_res[median_idx]
    large_match_res = match_res[large_idx]
    small_acc = small_match_res.sum()
    median_acc = median_match_res.sum()
    large_acc = large_match_res.sum()
    small_gt_num = len(small_match_res)
    median_gt_num = len(median_match_res)
    large_gt_num = len(large_match_res)
    all_acc[0] += small_acc.item()
    all_acc[1] += median_acc.item()
    all_acc[2] += large_acc.item()
    all_gt_num[0] += small_gt_num
    all_gt_num[1] += median_gt_num
    all_gt_num[2] += large_gt_num
    #print(small_acc, median_acc, large_acc, small_gt_num, median_gt_num, large_gt_num)
    #print((gt_gt_label.cuda() == pred_idx).sum().item(), len(gt_gt_label))
    #acc = (gt_gt_label.cuda() == pred_idx).sum()
    #all_acc += acc.item()
    #gt_num = len(gt_gt_label)
    #all_gt_num += gt_num

all_acc = np.array(all_acc).astype(float)
all_gt_num = np.array(all_gt_num).astype(float)
all_acc /= all_gt_num
print(all_acc)


confusion_matrix = torch.zeros(3, 80, 80)

for key in gt_predict_result.keys():
    #print(key)
    gt_pred = gt_predict_result[key]
    assigned_cate_ids = gt_assignment_res[key]
    this_size_tags = all_size_tags[key]
    #print(assigned_cate_ids)
    pred_idxes = torch.argmax(gt_pred, dim=1)
    # convert gt label to the embedding_idx
    for gt_cate_id, pred_idx, tag in zip(assigned_cate_ids, pred_idxes, this_size_tags):
        gt_idx = from_cate_id_to_embedding_id[gt_cate_id.item()]
        confusion_matrix[tag][gt_idx][pred_idx.item()] += 1

print(confusion_matrix)

normalized_mat = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True)
print(normalized_mat)

import pandas as pd
import numpy as np

temp = normalized_mat.numpy()
temp0 = pd.DataFrame(temp[0])
temp1 = pd.DataFrame(temp[1])
temp2 = pd.DataFrame(temp[2])
#temp.to_csv('normalized_mat.csv')
temp0.to_csv('normalized_mat_1_2time_zero_padding_small.csv')
temp1.to_csv('normalized_mat_1_2time_zero_padding_median.csv')
temp2.to_csv('normalized_mat_1_2time_zero_padding_large.csv')


# create legs feat
test  = clip.tokenize(f"a photo of legs").to(device)
cate_features = model.encode_text(test)
torch.save(cate_features.cpu(), 'legs_feat.pt')