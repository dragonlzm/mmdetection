# this script aims to visualize the lvis dataset, include the filtering with base

from email.mime import base
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
gt_content = json.load(open('/data/zhuoming/detection/lvis_v1/annotations/lvis_train_novel_base_subset.json'))

novel_id = []
base_id = []
for ele in gt_content['categories']:
    frequency = ele['frequency']
    category_id = ele['id']
    if frequency == 'c' or frequency == 'f':
        base_id.append(category_id)
    else:
        novel_id.append(category_id)

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
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
save_root = '/home/zhuoming/lvis_visualization'
proposal_path_root = '/data/zhuoming/detection/lvis_v1/clip_proposal/lvis_32_32_512'

for i, image_id in enumerate(from_image_id_to_annotation):
    # skip the image with novel instance
    if i < 337:
        continue
    
    #print(type(image_id))
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
    pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    all_proposals = torch.tensor(pregenerated_bbox['score'])
    
    # filter box base on the gt bbox
    base_gt_bbox = torch.tensor(from_image_id_to_annotation[image_id]['base'])
    #print('base_gt_bbox', base_gt_bbox.shape)
    if base_gt_bbox.shape[0] == 0:
        print(file_name)
        continue
    xyxy_gt = torch.cat([base_gt_bbox[:, 0].unsqueeze(dim=-1), base_gt_bbox[:, 1].unsqueeze(dim=-1), 
                         (base_gt_bbox[:, 0] + base_gt_bbox[:, 2]).unsqueeze(dim=-1), 
                         (base_gt_bbox[:, 1] + base_gt_bbox[:, 3]).unsqueeze(dim=-1)], dim=-1)

    real_iou = iou_calculator(xyxy_gt, all_proposals)
    max_iou_per_proposal = torch.max(real_iou, dim=0)[0]
    all_iou_idx = (max_iou_per_proposal < 0.3)
    remained_bbox = all_proposals[all_iou_idx]
    # sort the result base on the confidence score
    _, sorted_idx = torch.sort(remained_bbox[:, -1], descending=True)
    remained_bbox = remained_bbox[sorted_idx]
    
    print(remained_bbox.shape)
    # select top 50
    #sorted_proposal = sorted_proposal[:50]
    for box in remained_bbox[:50]:
        rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()

# if the iou of one gt bboxes with all proposals are equal to 0, then we should not draw any proposal
# for this bboxes