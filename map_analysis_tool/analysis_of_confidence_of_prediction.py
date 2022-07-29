# this script is aim to calculate the average confidence of the matched predictions
import json
import torch
import sys
import os
sys.path.append('../mmdetection')
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()

# load the prediction
#prediction_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/novel_results_trick.bbox.json'
#prediction_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json'
#prediction_path = '/data/zhuoming/detection/mask_rcnn_clip_classifier/results_base48_12.bbox.json'
prediction_path = '/data/zhuoming/detection/mask_rcnn_clip_classifier/results_novel17_12.bbox.json'

prediction_content = json.load(open(prediction_path))

# aggregate the prediction base on the image and base on the categories
from_image_id_to_pred = {}
for ele in prediction_content:
    image_id = ele['image_id']
    bbox = ele['bbox']
    score = ele['score']
    category_id = ele['category_id']
    if image_id not in from_image_id_to_pred:
        from_image_id_to_pred[image_id] = {}
    if category_id not in from_image_id_to_pred[image_id]:
        from_image_id_to_pred[image_id][category_id] = []
    bbox.append(score)
    from_image_id_to_pred[image_id][category_id].append(bbox)
    

# load the gt
#gt_path = '/data/zhuoming/detection/coco/annotations/instances_val2017.json'
gt_path = '/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'

gt_content = json.load(open(gt_path))

# aggregate the gt bboxes base on the image and base on the categories
from_image_id_to_gt = {}
for ele in gt_content['annotations']:
    image_id = ele['image_id']
    bbox = ele['bbox']
    category_id = ele['category_id']
    if image_id not in from_image_id_to_gt:
        from_image_id_to_gt[image_id] = {}
    if category_id not in from_image_id_to_gt[image_id]:
        from_image_id_to_gt[image_id][category_id] = []
    from_image_id_to_gt[image_id][category_id].append(bbox)

# for each image, for each category, calculate the iou, and find out the matched prediction
valid_prediction_num = 0
valid_prediction_conf = 0

for image_id in from_image_id_to_pred:
    # if the image does not have the gt bboxes, skip this image
    if image_id not in from_image_id_to_gt:
        continue
    for category_id in from_image_id_to_pred[image_id]:
        # if there is no gt annotation for this category, skip this categories
        if category_id not in from_image_id_to_gt[image_id]:
            continue
        pred_for_this_cate = torch.tensor(from_image_id_to_pred[image_id][category_id])
        gt_for_this_cate = torch.tensor(from_image_id_to_gt[image_id][category_id])
        
        iou = iou_calculator(pred_for_this_cate, gt_for_this_cate)
        #print('iou shape', iou.shape)
        max_iou_per_pred, max_iou_per_pred_idx = torch.max(iou, dim=-1)
        
        valid_prediction = pred_for_this_cate[max_iou_per_pred >= 0.5]
        if valid_prediction.shape[0] > 0:
            valid_prediction_num += valid_prediction.shape[0]
            valid_prediction_conf += torch.sum(valid_prediction[:, -1]).item()

print(valid_prediction_num, valid_prediction_conf, valid_prediction_conf / valid_prediction_num)


all_max_conf_score = 0
all_image_num = 0

# calculate the max confidence score average over image
for image_id in from_image_id_to_pred:
    if image_id not in from_image_id_to_gt:
        continue
    all_image_num += 1
    all_pred_per_img = []
    for category_id in from_image_id_to_pred[image_id]:
        pred_per_cate = torch.tensor(from_image_id_to_pred[image_id][category_id])
        all_pred_per_img.append(pred_per_cate)
    all_pred_per_img = torch.cat(all_pred_per_img, dim=0)
    all_pred_conf_per_img = all_pred_per_img[:, -1]
    #print(torch.max(all_pred_conf_per_img))
    all_max_conf_score += torch.max(all_pred_conf_per_img).item()

print(all_max_conf_score, all_image_num, all_max_conf_score / all_image_num)
    




# calculate the average confidence 
