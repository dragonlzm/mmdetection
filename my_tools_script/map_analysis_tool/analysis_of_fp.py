# this script aims analysis the ratio of different FP
# whether the fp come from the misclassification between the foreground and background
# or the fp come from the misclassification between different foreground categories 
import json
from pickletools import read_uint1
import torch
import sys
sys.path.append('../mmdetection')
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D

# aggregate the gt on each image base on the categoires
gt_path_1 = '/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'
gt_path_2 = '/data/zhuoming/detection/coco/annotations/instances_val2017_base48.json'

gt_anno_1 = json.load(open(gt_path_1))
gt_anno_2 = json.load(open(gt_path_2))

from_image_id_to_gtbboxes = {}

# collect the annotations for novel cate
for anno in gt_anno_1['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    if image_id not in from_image_id_to_gtbboxes:
        from_image_id_to_gtbboxes[image_id] = {}
    if category_id not in from_image_id_to_gtbboxes[image_id]:
        from_image_id_to_gtbboxes[image_id][category_id] = []
    from_image_id_to_gtbboxes[image_id][category_id].append(bbox)

# collect the annotations for base cate
for anno in gt_anno_2['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    if image_id not in from_image_id_to_gtbboxes:
        from_image_id_to_gtbboxes[image_id] = {}
    if category_id not in from_image_id_to_gtbboxes[image_id]:
        from_image_id_to_gtbboxes[image_id][category_id] = []
    from_image_id_to_gtbboxes[image_id][category_id].append(bbox)

# we aggregate the prediction on each image base on the categories
#pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/novel_results_65cates.bbox.json'

#pred_path = '/data/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x8_base48/base_results.bbox.json'
pred_path = '/data/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x8/base_results.bbox.json'

pred_res = json.load(open(pred_path))

from_image_id_to_prediction = {}

for pred in pred_res:
    image_id = pred['image_id']
    bbox = pred['bbox']
    category_id = pred['category_id']
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = {}
    if category_id not in from_image_id_to_prediction[image_id]:
        from_image_id_to_prediction[image_id][category_id] = []
    from_image_id_to_prediction[image_id][category_id].append(bbox)

iou_calculator = BboxOverlaps2D()
# for each image calculate the iou between the gt and prediction
all_final_prediction = []

def get_bbox_for_one_cate_and_the_rest(bbox_on_this_image, cate_id):
    if cate_id not in bbox_on_this_image:
        bbox_of_id = None
    else:
        bbox_of_id = bbox_on_this_image[cate_id]
    
    bbox_except_id = []
    for key in bbox_on_this_image:
        if key == cate_id:
            continue
        bbox_except_id += bbox_on_this_image[key]
    return bbox_of_id, bbox_except_id
    
all_bbox = 0
all_interclass_mis_cls = 0

for image_id in from_image_id_to_prediction:
    pred_on_this_image = from_image_id_to_prediction[image_id]
    # if image does not have gt
    if image_id not in from_image_id_to_gtbboxes:
        # do nothing to avoid the problem
        continue
    # we do not need to handle the situation where there is gt but no prediction
    gt_on_this_image = from_image_id_to_gtbboxes[image_id]
    for cate_id in pred_on_this_image:
        # split the gt into two section
        gt_of_id, gt_except_id = get_bbox_for_one_cate_and_the_rest(gt_on_this_image, cate_id)
        if len(gt_except_id) == 0:
            gt_of_id = torch.tensor(gt_of_id)
            iou_in = iou_calculator(all_pred_bboxes, gt_of_id)
            max_iou_in_per_pred_val, max_iou_in_per_pred_idx = torch.max(iou_in, dim=-1)
            
            miss_indx = (max_iou_in_per_pred_val < 0.5)
            all_bbox += torch.sum(miss_indx).item()
            continue
        # get the predicted bbox
        all_pred_bboxes = torch.tensor(pred_on_this_image[cate_id])
        gt_except_id = torch.tensor(gt_except_id)
        # calculate the iou between the predition and the gt of other categories
        iou_except = iou_calculator(all_pred_bboxes, gt_except_id)
        max_iou_except_per_pred_val, max_iou_except_per_pred_idx = torch.max(iou_except, dim=-1)
        
        # if there is prediction but no gt of this cate
        # directly calculate the iou between all prediction and all gts(with different categories)
        if gt_of_id == None:
            mis_class_idx = (max_iou_except_per_pred_val > 0.5)
            mis_cls_num = torch.sum(mis_class_idx)
            all_interclass_mis_cls += mis_cls_num.item()
            all_bbox += mis_class_idx.shape[0]
            continue
        
        gt_of_id = torch.tensor(gt_of_id)
        iou_in = iou_calculator(all_pred_bboxes, gt_of_id)
        max_iou_in_per_pred_val, max_iou_in_per_pred_idx = torch.max(iou_in, dim=-1)
        
        miss_indx = (max_iou_in_per_pred_val < 0.5)
        mis_class_idx = (max_iou_except_per_pred_val > 0.5)
        mis_cls_num = torch.sum((miss_indx & mis_class_idx))
        
        all_interclass_mis_cls += mis_cls_num.item()
        all_bbox += torch.sum(miss_indx).item()

print('all_bbox', all_bbox, 'all_interclass_mis_cls', all_interclass_mis_cls, 'precentage', all_interclass_mis_cls / all_bbox)