# this script aims to calculate the statistic number of the CLIP
# the average iou, overall overlap area, overlap ratio

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
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_0_8000.json'))

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
proposal_path_root = ['/data/zhuoming/detection/coco/clip_proposal/32_32_512', 
                      '/data/zhuoming/detection/coco/rpn_proposal/mask_rcnn_r50_fpn_2x_coco_2gpu_base48_reg_class_agno',
                      '/data/zhuoming/detection/coco/clip_proposal/32_32_512_imagenet1762/']

for root in proposal_path_root:
    all_iou_for_novel = 0
    all_overlap_on_novel = 0
    image_with_novel = 0
    
    all_iou_for_base = 0
    all_overlap_on_base = 0
    image_with_base = 0
    for i, image_id in enumerate(from_image_id_to_annotation):
        base_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['base'])
        novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
        if base_gt_bboxes.shape[0] == 0 and novel_gt_bboxes.shape[0] == 0:
            continue
        #download the image
        file_name = from_image_id_to_image_info[image_id]['coco_url'].split('/')[-1]

        pregenerate_prop_path = os.path.join(root, '.'.join(file_name.split('.')[:-1]) + '.json')
        pregenerated_bbox = json.load(open(pregenerate_prop_path))
        
        # read the proposal directly from the proposal file
        all_proposals = torch.tensor(pregenerated_bbox['score'])
        
        # for extract proposal from the feature file 
        # all_proposals = torch.tensor(pregenerated_bbox['bbox'])
        # ## scale the bbox back to the original size
        # img_metas = pregenerated_bbox['img_metas']
        # pre_extract_scale_factor = np.array(img_metas['scale_factor']+[1,1]).astype(np.float32)
        # # all_proposals: [562.7451782226562, 133.49032592773438, 653.2548217773438, 314.5096740722656, 0.9763965010643005, 461.0]
        # all_proposals = all_proposals / pre_extract_scale_factor
        
        # find the matched bbox for gt bbox "novel"
        if novel_gt_bboxes.shape[0] != 0:
            # convert the gt for the novel
            novel_gt_bboxes[:, 2] = novel_gt_bboxes[:, 0] + novel_gt_bboxes[:, 2] 
            novel_gt_bboxes[:, 3] = novel_gt_bboxes[:, 1] + novel_gt_bboxes[:, 3] 
            proposal_with_novel_iou = iou_calculator(all_proposals[:, :4], novel_gt_bboxes)
            accumulate_proposal_with_novel_iou = torch.mean(proposal_with_novel_iou[proposal_with_novel_iou != 0])
            if torch.isnan(accumulate_proposal_with_novel_iou):
                #print(proposal_with_novel_iou[proposal_with_novel_iou != 0])
                accumulate_proposal_with_novel_iou = 0
            all_iou_for_novel += accumulate_proposal_with_novel_iou
            image_with_novel += 1
            
            # calculate the overall overlap with the gt bboxes
            overlap_on_novel_gt = iou_calculator(novel_gt_bboxes, all_proposals[:, :4], 'iof')
            accumulate_overlap_on_novel_gt = torch.mean(overlap_on_novel_gt[overlap_on_novel_gt != 0])
            if torch.isnan(accumulate_overlap_on_novel_gt):
                #print(overlap_on_novel_gt[overlap_on_novel_gt != 0])
                accumulate_overlap_on_novel_gt = 0
            all_overlap_on_novel += accumulate_overlap_on_novel_gt
            
        
        if base_gt_bboxes.shape[0] != 0:
            # convert the gt for the base
            base_gt_bboxes[:, 2] = base_gt_bboxes[:, 0] + base_gt_bboxes[:, 2] 
            base_gt_bboxes[:, 3] = base_gt_bboxes[:, 1] + base_gt_bboxes[:, 3] 
            proposal_with_base_iou = iou_calculator(all_proposals[:, :4], base_gt_bboxes)
            accumulate_proposal_with_base_iou = torch.mean(proposal_with_base_iou[proposal_with_base_iou != 0])
            all_iou_for_base += accumulate_proposal_with_base_iou
            image_with_base += 1 
            
            # calculate the overall overlap with the gt bboxes
            overlap_on_base_gt = iou_calculator(base_gt_bboxes, all_proposals[:, :4], 'iof')
            accumulate_overlap_on_base_gt = torch.mean(overlap_on_base_gt[overlap_on_base_gt != 0])
            all_overlap_on_base += accumulate_overlap_on_base_gt
            
        
    print(root, 'avg iou for novel:', all_iou_for_novel / image_with_novel, 'avg iou for base:', all_iou_for_base / image_with_base,
          'avg overlap on gt for novel:', all_overlap_on_novel / image_with_novel, 'avg overlap on gt for base:', all_overlap_on_base / image_with_base)
