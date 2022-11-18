# this script aims to visualize the coco dataset, include the filtering with base

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
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017.json'))

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
save_root = '/home/zhuoming/coco_visualization_most_matched_base_and_novel'
proposal_path_root = '/data/zhuoming/detection/coco/clip_proposal/32_32_512'

# save_root = '/home/zhuoming/coco_visualization_most_matched_base_and_novel_rpn_proposal'
# proposal_path_root = '/data/zhuoming/detection/coco/rpn_proposal/mask_rcnn_r50_fpn_2x_coco_2gpu_base48_reg_class_agno'

count = 0
for i, image_id in enumerate(from_image_id_to_annotation):
    base_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['base'])
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
    pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    # for loading the proposal file version
    all_proposals = torch.tensor(pregenerated_bbox['score'])
    
    # for loading the feature file version
    # all_proposals = torch.tensor(pregenerated_bbox['bbox'])
    # ## scale the bbox back to the original size
    # img_metas = pregenerated_bbox['img_metas']
    # pre_extract_scale_factor = np.array(img_metas['scale_factor']+[1,1]).astype(np.float32)
    # # all_proposals: [562.7451782226562, 133.49032592773438, 653.2548217773438, 314.5096740722656, 0.9763965010643005, 461.0]
    # all_proposals = all_proposals / pre_extract_scale_factor
    
    # find the matched bbox for gt bbox "novel"
    if novel_gt_bboxes.shape[0] != 0:
        match = False
        #all_predicted_cate = all_proposals[:, -1]
        for novel_bbox in novel_gt_bboxes:
            xyxy_gt = torch.tensor([[novel_bbox[0], novel_bbox[1], novel_bbox[0] + novel_bbox[2], 
                                novel_bbox[1] + novel_bbox[3]]])
            real_iou = iou_calculator(xyxy_gt, all_proposals[:, :4])
            # leave the iou value only when the iou larger than 0
            iou_idx_over_zero = (real_iou > 0)
            #real_iou = real_iou[real_iou > 0]
            # select the top 10 for each gt bboxes
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            elif torch.sum(iou_idx_over_zero) < 10:
                remain_proposal = all_proposals[iou_idx_over_zero.squeeze(dim=0)]
            else:
                value, idx = torch.topk(real_iou, 10)
                remain_proposal = all_proposals[idx.squeeze(dim=0)]
            
            for box in remain_proposal:
                rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)    

    # if base_gt_bboxes.shape[0] != 0:
    #     match = False
    #     #all_predicted_cate = all_proposals[:, -1]
    #     for base_bbox in base_gt_bboxes:
    #         xyxy_gt = torch.tensor([[base_bbox[0], base_bbox[1], base_bbox[0] + base_bbox[2], 
    #                             base_bbox[1] + base_bbox[3]]])
    #         real_iou = iou_calculator(xyxy_gt, all_proposals[:, :4])
    #         # leave the iou value only when the iou larger than 0
    #         iou_idx_over_zero = (real_iou > 0)
    #         #real_iou = real_iou[real_iou > 0]
    #         # select the top 10 for each gt bboxes
    #         if torch.sum(iou_idx_over_zero) == 0:
    #             continue
    #         elif torch.sum(iou_idx_over_zero) < 10:
    #             remain_proposal = all_proposals[iou_idx_over_zero.squeeze(dim=0)]
    #         else:
    #             value, idx = torch.topk(real_iou, 10)
    #             remain_proposal = all_proposals[idx.squeeze(dim=0)]
            
    #         for box in remain_proposal:
    #             rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
    #             ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()

