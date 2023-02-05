# this script are used in visualize the boxes selected by the CLIP# this script aims to visualize the coco dataset, include the filtering with base

import json
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()

# load the gt bboxes
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json'))
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017.json'))
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))

all_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']
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
novel_name_idx = [from_name_to_idx[name] for name in novel_names]

novel_id = []
base_id = []
for ele in gt_content['categories']:
    category_id = ele['id']
    name = ele['name']
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
save_root = '/home/zhuoming/coco_visualization_vit_clip_selected'
#proposal_path_root = '/data/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/random'
#proposal_path_root = '/home/zhuoming/detectron_proposal1'
proposal_path_root = '/home/zhuoming/detectron_proposal2'

count = 0
for i, image_id in enumerate(from_image_id_to_annotation):
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
    if novel_gt_bboxes.shape[0] == 0:
        continue
    else:
        count += 1
    if count > 100:
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
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    # print the base
    for box in from_image_id_to_annotation[image_id]['base']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    # print the novel
    for box in from_image_id_to_annotation[image_id]['novel']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # load the proposal 
    pregenerate_prop_path = os.path.join(proposal_path_root, (file_name + '_final_pred.json'))
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    all_proposals = torch.tensor(pregenerated_bbox['box'])
    # load the clip score of each bboxes
    clip_score_path = os.path.join(proposal_path_root, (file_name + '_clip_pred.json'))
    all_clip_scores = json.load(open(clip_score_path))
    all_clip_scores = torch.tensor(all_clip_scores['score'])
    
    # find out which bboxes is regarded as the bboxes contain novel object by CLIP
    value, max_cate_idx = torch.max(all_clip_scores, dim=-1)
    novel_idx = torch.tensor([i for i, ele in enumerate(max_cate_idx) if ele in novel_name_idx])
    if len(novel_idx) != 0:
        all_candicate_bboxes = all_proposals[novel_idx]
        for candicate_bbox in all_candicate_bboxes:
            rect = patches.Rectangle((candicate_bbox[0], candicate_bbox[1]), candicate_bbox[2] - candicate_bbox[0], candicate_bbox[3] - candicate_bbox[1],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)   

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()
