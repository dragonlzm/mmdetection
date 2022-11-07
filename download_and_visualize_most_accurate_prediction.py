# this script aims to visualize the top K prediction(sorted by confidence score)
import json
import os
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()


# load the prediction file
file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight/base_and_novel.bbox.json'

pred_content = json.load(open(file_path))
# aggregate the predition base on the image
from_image_id_to_prediction = {}
for res in pred_content:
    image_id = res['image_id']
    bbox = res['bbox']
    score = res['score']
    category_id = res['category_id']
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = {'bboxes':[], 'scores':[], 'category_id':[]}
    from_image_id_to_prediction[image_id]['bboxes'].append(bbox)
    from_image_id_to_prediction[image_id]['scores'].append(score)
    from_image_id_to_prediction[image_id]['category_id'].append(category_id)

# load the gt bboxes
gt_anno_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'
from_image_id_to_image_file_name = {}

# aggregate the gt bboxes base on the image
gt_content = json.load(open(gt_anno_file))

base_cates_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

from_cate_name_to_cate_id = {anno['name']: anno['id'] for anno in gt_content['categories']}
base_cates_ids = [from_cate_name_to_cate_id[name] for name in base_cates_name]

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    if cate_id in base_cates_ids:
        from_image_id_to_annotation[image_id]['base'].append(bbox)
    else:
        from_image_id_to_annotation[image_id]['novel'].append(bbox)
        
# collect the image info
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

count = 0
save_root = '/home/zhuoming/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_visualization'
for i, image_id in enumerate(from_image_id_to_annotation):
    # print the prediction
    all_prediction = from_image_id_to_prediction[image_id]['bboxes']
    base_gt_bboxes = from_image_id_to_annotation[image_id]['base']
    novel_gt_bboxes = from_image_id_to_annotation[image_id]['novel']
    
    # we only visualize the image with novel instance
    if novel_gt_bboxes.shape[0] == 0:
        continue
    else:
        count += 1
    if count > 50:
        break

    # download the image
    url = from_image_id_to_image_info[image_id]['coco_url']
    file_name = from_image_id_to_image_info[image_id]['file_name']
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
    # for box in from_image_id_to_annotation[image_id]['base']:
    #     rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
    #     ax.add_patch(rect)
    # print the novel
    # for box in from_image_id_to_annotation[image_id]['novel']:
    #     rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
    #     ax.add_patch(rect)
    
    # for novel
    # find the matched bbox for gt bbox "novel"
    if novel_gt_bboxes.shape[0] != 0:
        match = False
        #all_predicted_cate = all_proposals[:, -1]
        for novel_bbox in novel_gt_bboxes:
            xyxy_gt = torch.tensor([[novel_bbox[0], novel_bbox[1], novel_bbox[0] + novel_bbox[2], 
                                novel_bbox[1] + novel_bbox[3]]])
            real_iou = iou_calculator(xyxy_gt, all_prediction[:, :4])
            # leave the iou value only when the iou larger than 0
            iou_idx_over_zero = (real_iou > 0)
            #real_iou = real_iou[real_iou > 0]
            # select the top 10 for each gt bboxes
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            elif torch.sum(iou_idx_over_zero) < 10:
                remain_proposal = all_prediction[iou_idx_over_zero.squeeze(dim=0)]
            else:
                value, idx = torch.topk(real_iou, 10)
                remain_proposal = all_prediction[idx.squeeze(dim=0)]
            
            for box in remain_proposal:
                rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)        
    
    if base_gt_bboxes.shape[0] != 0:
        match = False
        #all_predicted_cate = all_proposals[:, -1]
        for base_bbox in base_gt_bboxes:
            xyxy_gt = torch.tensor([[base_bbox[0], base_bbox[1], base_bbox[0] + base_bbox[2], 
                                base_bbox[1] + base_bbox[3]]])
            real_iou = iou_calculator(xyxy_gt, all_prediction[:, :4])
            # leave the iou value only when the iou larger than 0
            iou_idx_over_zero = (real_iou > 0)
            #real_iou = real_iou[real_iou > 0]
            # select the top 10 for each gt bboxes
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            elif torch.sum(iou_idx_over_zero) < 10:
                remain_proposal = all_prediction[iou_idx_over_zero.squeeze(dim=0)]
            else:
                value, idx = torch.topk(real_iou, 10)
                remain_proposal = all_prediction[idx.squeeze(dim=0)]
            
            for box in remain_proposal:
                rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)      

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()