import os

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

'''
# prepare the coco cate text embeddings
json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'

coco_json = json.load(open(json_file_path))

all_cate_feat = []
for cate_info in coco_json['categories']:
    name = cate_info['name']
    # handle the space in the cate_name
    if ' ' in name:
        name = name.replace(' ', '_')
    tokenized_result = clip.tokenize(f"a photo of a {name}").to(device)
    with torch.no_grad():
        cate_features = model.encode_text(tokenized_result)
    #print(torch.norm(cate_features-single_feat))
    all_cate_feat.append(cate_features)

all_cate_feat = torch.cat(all_cate_feat, dim=0)
print(all_cate_feat.shape)

#torch.save(all_cate_feat, 'coco_name_feat_combined.pt')
torch.save(all_cate_feat, 'coco_name_feat_space_replaced.pt')'''

'''
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
            # enlarge the image with 10% for each cord
            #x_start_pos = math.floor(x)
            #y_start_pos = math.floor(y)
            x_start_pos = math.floor(max(x-0.1*w, 0))
            y_start_pos = math.floor(max(y-0.1*h, 0))
            #x_end_pos = math.ceil(x+w)
            #y_end_pos = math.ceil(y+h)
            x_end_pos = math.ceil(min(x+1.1*w, img_w-1))
            y_end_pos = math.ceil(min(y+1.1*h, img_h-1))
            now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]
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
torch.save(all_feature_res, 'val_img_enlarged_gt_rand_feat_res.pt')
torch.save(all_assigned_result, 'val_img_enlarged_gt_rand_all_assigned_result.pt')'''


# predict the categories (calculate the entropy)
text_embedding = torch.load('imagenet_name_feat_combined.pt')
#image_embedding = torch.load('100img_gt_rand_feat_res.pt')

#text_embedding = torch.load('coco_name_feat_space_replaced.pt')
#text_embedding = torch.load('coco_name_feat_combined.pt')
#image_embedding = torch.load('val_img_gt_rand_feat_res.pt')
image_embedding = torch.load('val_img_enlarged_gt_rand_feat_res.pt')

text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

gt_predict_result = {}
rand_predict_result = {}

all_gt_entropy = []
all_rand_entropy = []

total_gt_entry = 0
total_rand_entry = 0

for key in image_embedding.keys():
    image_features = image_embedding[key].cuda()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_embedding.T).softmax(dim=-1)
    #print(similarity.shape)
    #break
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
        #print(log_result)
        gt_entropy = (log_result * gt_pred).sum() 
        #gt_entropy /= gt_pred.shape[0]
        total_gt_entry += gt_pred.shape[0]
        #print(gt_entropy)
        all_gt_entropy.append(gt_entropy.item())
    # calculate the random entropy
    # deal with the 0 in the rand_pred
    rand_pred[rand_pred == 0] = 1e-5
    log_result = - torch.log(rand_pred)
    rand_entropy = (log_result * rand_pred).sum()
    #rand_entropy /= rand_pred.shape[0]
    total_rand_entry += rand_pred.shape[0]
    #print(log_result)
    #print(rand_entropy)
    all_rand_entropy.append(rand_entropy.item())

#final_gt_entro = sum(all_gt_entropy) / len(all_gt_entropy)
#final_rand_entro = sum(all_rand_entropy) / len(all_rand_entropy)

final_gt_entro = sum(all_gt_entropy) / total_gt_entry
final_rand_entro = sum(all_rand_entropy) / total_rand_entry

print('final_gt_entro: ', final_gt_entro, 'final_rand_entro', final_rand_entro)
#values, indices = similarity[0].topk(5)

#torch.save(gt_predict_result, 'coco80_gt_predict_result.pt')
#torch.save(rand_predict_result, 'coco80_rand_predict_result.pt')

torch.save(gt_predict_result, 'coco80_gt_predict_result_enlarged.pt')
torch.save(rand_predict_result, 'coco80_rand_predict_result_enlarged.pt')

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
    
'''
# predict acc with assignment result
gt_predict_result = torch.load('coco80_gt_predict_result_enlarged.pt')
gt_assignment_res = torch.load('val_img_enlarged_gt_rand_all_assigned_result.pt')

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
temp.to_csv('normalized_mat.csv')'''