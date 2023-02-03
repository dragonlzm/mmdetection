# this script aims to add fn predicted bbox
# if the gt bboxes do not have iou over 0.5 with all predicted bbox, add this gt bboxes to the prediction
import json
import torch
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
pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/novel_results_65cates.bbox.json'
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
for image_id in from_image_id_to_prediction:
    pred_on_this_image = from_image_id_to_prediction[image_id]
    # if image does not have gt
    if image_id not in from_image_id_to_gtbboxes:
        continue
    # we do not need to handle the situation where there is gt but no prediction
    gt_on_this_image = from_image_id_to_gtbboxes[image_id]
    for cate_id in pred_on_this_image:
        # if there is prediction but no gt of this cate 
        # set the confidence score to 0
        if cate_id not in gt_on_this_image:
            continue
        # for the situation that there are prediction and gt at the same time 
        all_gt_bboxes = torch.tensor(gt_on_this_image[cate_id])
        all_pred_bboxes = torch.tensor(pred_on_this_image[cate_id])
        # convert xywh to xyxy
        all_gt_bboxes[:, 2] = all_gt_bboxes[:, 0] + all_gt_bboxes[:, 2]
        all_gt_bboxes[:, 3] = all_gt_bboxes[:, 1] + all_gt_bboxes[:, 3]
        all_pred_bboxes[:, 2] = all_pred_bboxes[:, 0] + all_pred_bboxes[:, 2]
        all_pred_bboxes[:, 3] = all_pred_bboxes[:, 1] + all_pred_bboxes[:, 3]
        real_iou = iou_calculator(all_gt_bboxes, all_pred_bboxes)
        max_iou_per_gt_val, max_iou_per_gt_idx = torch.max(real_iou, dim=-1)
        # convert xyxy back to xywh
        all_gt_bboxes[:, 2] = all_gt_bboxes[:, 2] - all_gt_bboxes[:, 0]
        all_gt_bboxes[:, 3] = all_gt_bboxes[:, 3] - all_gt_bboxes[:, 1]
        for gt_bbox, max_iou_val in zip(all_gt_bboxes, max_iou_per_gt_val):
            if max_iou_val < 0.5:
                info = {'image_id': image_id, 'bbox': gt_bbox.tolist(), 'score': 1.0, 'category_id': cate_id}
                pred_res.append(info)

print(len(pred_res))
file = open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/add_fn_prediction.json', 'w')
file.write(json.dumps(pred_res))
file.close()