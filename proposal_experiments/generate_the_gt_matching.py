# this script aims to find out the most matched proposal for each novel gt (by iou)
# and save the respective gt label and the clip conf distribution

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
save_root = '/home/zhuoming/detectron_proposal2'
#proposal_path_root = '/data/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/random'
#proposal_path_root = '/home/zhuoming/detectron_proposal1'
proposal_path_root = '/home/zhuoming/detectron_proposal2'

count = 0
for i, image_id in enumerate(from_image_id_to_annotation):
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
    file_name = from_image_id_to_image_info[image_id]['coco_url'].split('/')[-1]
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # of the image do not have the novel gt, then handle the special conditions
    if novel_gt_bboxes.shape[0] == 0:
        all_target_proposal_idx = []
        # you need to save the gt label into the idx format
        all_target_proposal_gt_idx = []
        all_target_proposal_clip_distri = []
    else:
        # load the final predict 
        #pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
        pregenerate_prop_path = os.path.join(proposal_path_root, (file_name + '_final_pred.json'))
        pregenerated_bbox = json.load(open(pregenerate_prop_path))
        #all_proposals = torch.tensor(pregenerated_bbox['bbox'])
        all_proposals = torch.tensor(pregenerated_bbox['box'])
        
        # load the clip score
        clip_path = os.path.join(proposal_path_root, (file_name + '_clip_pred.json'))
        clip_content = json.load(open(clip_path))        
        all_clip_score = torch.tensor(clip_content['score'])
        
        # load the proposal
        proposal_path = os.path.join(proposal_path_root, (file_name + '.json'))
        proposal_content = json.load(open(proposal_path))        
        all_objectness_score = torch.tensor(proposal_content['score'])
        
        all_target_proposal_idx = []
        all_target_proposal_gt_idx = []
        all_target_proposal_clip_distri = []
        
        for novel_box in from_image_id_to_annotation[image_id]['novel']:
            novel_bbox_cate_id = novel_box[-1]
            novel_cate_name = from_gt_id_to_name[novel_bbox_cate_id]
            if novel_cate_name not in novel_names:
                continue
            novel_cate_idx = from_name_to_idx[novel_cate_name]
            
            #print('novel_bbox', novel_bbox)
            xyxy_gt = torch.tensor([[novel_box[0], novel_box[1], novel_box[0] + novel_box[2], 
                                novel_box[1] + novel_box[3]]])
            #print('xyxy_gt', xyxy_gt)
            real_iou = iou_calculator(xyxy_gt, all_proposals[:, :4])
            iou_idx_over_zero = (real_iou > 0)
            # if there is no proposal has iou over 0.5 with the gt bboxes then skip the bboxes
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            else:
                value, idx = torch.max(real_iou, dim=-1)
                if value < 0.5:
                    continue
                remain_proposal = all_proposals[idx].squeeze(dim=0)
                # prepare the clip score
                clip_score_for_proposal = all_clip_score[idx]
                softmax_score = softmax_fun(clip_score_for_proposal)
                max_clip_score_val, max_clip_score_idx = torch.max(clip_score_for_proposal, dim=-1)
                pred_cate_name = from_idx_to_name[max_clip_score_idx.item()]
                all_target_proposal_idx.append(idx.item())
                # you need to save the gt label into the idx format
                all_target_proposal_gt_idx.append(novel_cate_idx)
                #print('softmax_score', softmax_score.shape)
                all_target_proposal_clip_distri.append(softmax_score.squeeze(dim=0).tolist())
        #print('all_target_proposal_idx', all_target_proposal_idx, 'all_target_proposal_gt_idx', all_target_proposal_gt_idx, 'all_target_proposal_clip_distri', all_target_proposal_clip_distri)
    
    # save the result
    all_result = {'all_target_proposal_idx':all_target_proposal_idx, 'all_target_proposal_gt_idx':all_target_proposal_gt_idx, 'all_target_proposal_clip_distri':all_target_proposal_clip_distri}
    replaced_file_name = file_name + 'match_gt.json'
    file = open(os.path.join(save_root, replaced_file_name), 'w')
    file.write(json.dumps(all_result))
    file.close()
