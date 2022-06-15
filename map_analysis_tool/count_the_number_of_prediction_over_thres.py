# this script aims to count the number of preditions which iou with any gt over the threshold 
import json
import torch
import sys
sys.path.append('../mmdetection')
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D

pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results.bbox.json'))

#pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json'))
# aggregate the prediction base on the image_id and then aggregate the infomation according to
# the categories

# both preditions and the gt annotations are xywh format
from_image_id_to_prediction = {}

# aggreate the prediction
for pred in pred_res:
    image_id = pred['image_id']
    bbox = pred['bbox']
    category_id = pred['category_id']
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = {}
    if category_id not in from_image_id_to_prediction[image_id]:
        from_image_id_to_prediction[image_id][category_id] = []
    from_image_id_to_prediction[image_id][category_id].append(bbox)

# aggregate the annotation
gt_file = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'))
from_image_id_to_gt_anno = {}
for anno in gt_file['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    if image_id not in from_image_id_to_gt_anno:
        from_image_id_to_gt_anno[image_id] = {}
    if category_id not in from_image_id_to_gt_anno[image_id]:
        from_image_id_to_gt_anno[image_id][category_id] = []
    from_image_id_to_gt_anno[image_id][category_id].append(bbox)
    
    
iou_calculator = BboxOverlaps2D()
# over_threshold_prediction = 0

# # match the prediction
# for image_id in from_image_id_to_prediction:
#     if image_id not in from_image_id_to_gt_anno:
#         continue
#     for category_id in from_image_id_to_prediction[image_id]:
#         if category_id not in from_image_id_to_gt_anno[image_id]:
#             continue
#         all_pred_for_this_cate = torch.tensor(from_image_id_to_prediction[image_id][category_id])
#         all_gt_for_this_cate = torch.tensor(from_image_id_to_gt_anno[image_id][category_id])
        
#         all_pred_for_this_cate[:, 2] = all_pred_for_this_cate[:, 2] + all_pred_for_this_cate[:, 0]
#         all_pred_for_this_cate[:, 3] = all_pred_for_this_cate[:, 3] + all_pred_for_this_cate[:, 1]
#         all_gt_for_this_cate[:, 2] = all_gt_for_this_cate[:, 2] + all_gt_for_this_cate[:, 0]
#         all_gt_for_this_cate[:, 3] = all_gt_for_this_cate[:, 3] + all_gt_for_this_cate[:, 1]  
#         the_iou_per_pred = iou_calculator(all_pred_for_this_cate, all_gt_for_this_cate)
#         max_val, _ = torch.max(the_iou_per_pred, dim=1)
#         over_thres_idx = (max_val > 0.5)
#         over_thres_num = over_thres_idx.sum()
#         over_threshold_prediction += over_thres_num.item()
        
# print('number of over threshold:', over_threshold_prediction)
        

over_threshold_match_gt = 0
# match the prediction
for image_id in from_image_id_to_gt_anno:
    for category_id in from_image_id_to_gt_anno[image_id]:
        if category_id not in from_image_id_to_prediction[image_id]:
            continue
        all_pred_for_this_cate = torch.tensor(from_image_id_to_prediction[image_id][category_id])
        all_gt_for_this_cate = torch.tensor(from_image_id_to_gt_anno[image_id][category_id])
        
        all_pred_for_this_cate[:, 2] = all_pred_for_this_cate[:, 2] + all_pred_for_this_cate[:, 0]
        all_pred_for_this_cate[:, 3] = all_pred_for_this_cate[:, 3] + all_pred_for_this_cate[:, 1]
        all_gt_for_this_cate[:, 2] = all_gt_for_this_cate[:, 2] + all_gt_for_this_cate[:, 0]
        all_gt_for_this_cate[:, 3] = all_gt_for_this_cate[:, 3] + all_gt_for_this_cate[:, 1]  
        the_iou_per_pred = iou_calculator(all_gt_for_this_cate, all_pred_for_this_cate)
        max_val, _ = torch.max(the_iou_per_pred, dim=1)
        over_thres_idx = (max_val > 0.5)
        over_thres_num = over_thres_idx.sum()
        over_threshold_match_gt += over_thres_num.item()
        
print('number of over threshold:', over_threshold_match_gt)
        
