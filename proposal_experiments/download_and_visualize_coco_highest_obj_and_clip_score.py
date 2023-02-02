# this version draw a img for each novel categories bboxes
# this script aims to visualize the coco dataset, include the filtering with base

import json
import torch
from torch import nn
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()

softmax_fun = nn.Softmax(dim=1)
sigmoid_fun = nn.Sigmoid()
# load the gt bboxes
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json'))
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017.json'))
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))


all_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')
base_names = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_names = ('airplane', 'bus', 'cat', 'dog', 'cow', 
            'elephant', 'umbrella', 'tie', 'snowboard', 
            'skateboard', 'cup', 'knife', 'cake', 'couch', 
            'keyboard', 'sink', 'scissors')

from_name_to_idx = {name:i for i, name in enumerate(all_names)}
from_idx_to_name = {i:name for i, name in enumerate(all_names)}

from_gt_id_to_name = {}
novel_id = []
base_id = []
for ele in gt_content['categories']:
    category_id = ele['id']
    name = ele['name']
    from_gt_id_to_name[category_id] = name
    if name in base_names:
        base_id.append(category_id)
    elif name in novel_names:
        novel_id.append(category_id)

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    bbox.append(cate_id)
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    if cate_id in base_id:
        from_image_id_to_annotation[image_id]['base'].append(bbox)
    else:
        from_image_id_to_annotation[image_id]['novel'].append(bbox)

# collect the image info
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

# load the proposal and print the image
#save_root = '/home/zhuoming/coco_visualization_most_matched'
save_root = '/home/zhuoming/coco_visualization_vit_most_matched_per_novel'
#proposal_path_root = '/data/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/random'
#proposal_path_root = '/home/zhuoming/detectron_proposal1'
proposal_path_root = '/home/zhuoming/detectron_proposal2'

count = 0
for i, image_id in enumerate(from_image_id_to_annotation):
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
    count += 1
    if count > 20:
        break

    #download the image
    url = from_image_id_to_image_info[image_id]['coco_url']
    file_name = from_image_id_to_image_info[image_id]['coco_url'].split('/')[-1]
    save_path = os.path.join(save_root, file_name)
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    im = np.array(Image.open(save_path), dtype=np.uint8)
    
    # load the final predict 
    pregenerate_prop_path = os.path.join(proposal_path_root, (file_name + '_final_pred.json'))
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    all_proposals = torch.tensor(pregenerated_bbox['box'])
    
    # load the clip score
    clip_path = os.path.join(proposal_path_root, (file_name + '_clip_pred.json'))
    clip_content = json.load(open(clip_path))        
    all_clip_score = torch.tensor(clip_content['score'])
    
    # load the proposal
    proposal_path = os.path.join(proposal_path_root, (file_name + '.json'))
    proposal_content = json.load(open(proposal_path))        
    all_objectness_score = torch.tensor(proposal_content['score'])
    
    # merge the clip score and the
    softmax_score = softmax_fun(all_clip_score)
    softmax_score = softmax_score.squeeze(dim=0)
    max_softmax_score, _ = torch.max(softmax_score, dim=-1)

    all_objectness_score = sigmoid_fun(all_objectness_score)
    merged_score = all_objectness_score + max_softmax_score
    
    # show the image
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    
    # draw the novel bboxes
    if novel_gt_bboxes.shape[0] != 0:
        for novel_box in from_image_id_to_annotation[image_id]['novel']:
            rect = patches.Rectangle((novel_box[0], novel_box[1]),novel_box[2],novel_box[3],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

    # draw the bboxes with the highest confidence
    _, selected_idx = torch.topk(merged_score, 3)
    selected_proposals = all_proposals[selected_idx]

    for proposal in selected_proposals:
        rect = patches.Rectangle((proposal[0], proposal[1]),proposal[2]-proposal[0],proposal[3]-proposal[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)
    save_file_name = file_name
    plt.savefig(os.path.join(print_path, save_file_name))
    plt.close()