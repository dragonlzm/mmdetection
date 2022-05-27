# this script aims to reassign the label of the predicted bboxes as the
# label of the gt bboxes which has the max iou with this bboxes
import json
import torch
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D

# we using all annotations of 65 categories to assign the result
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
        from_image_id_to_gtbboxes[image_id] = []
    from_image_id_to_gtbboxes[image_id].append((bbox,category_id))

# collect the annotations for base cate
for anno in gt_anno_2['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    if image_id not in from_image_id_to_gtbboxes:
        from_image_id_to_gtbboxes[image_id] = []
    from_image_id_to_gtbboxes[image_id].append((bbox,category_id))

# load the prediction
pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/novel_results_65cates.bbox.json'

pred_res = json.load(open(pred_path))

from_image_id_to_prediction = {}

for pred in pred_res:
    image_id = pred['image_id']
    bbox = pred['bbox']
    category_id = pred['category_id']
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = []
    from_image_id_to_prediction[image_id].append((bbox,category_id))

iou_calculator = BboxOverlaps2D()
# for each image calculate the iou between the gt and prediction
all_final_prediction = []
for image_id in from_image_id_to_prediction:
    predicted_res = from_image_id_to_prediction[image_id]
    # if there is no gt annotation on the image
    if image_id not in from_image_id_to_gtbboxes:
        for pred_bbox, pred_label in predicted_res:
            info = {'image_id': image_id, 'bbox': pred_bbox.tolist(), 'score': 0.0, 'category_id': pred_label.item()}
            all_final_prediction.append(info)
        continue
    
    predicted_bboxes = torch.tensor([ele[0] for ele in predicted_res])
    # convert from xywh to xyxy
    predicted_bboxes[:, 2] = predicted_bboxes[:, 2] + predicted_bboxes[:, 0]
    predicted_bboxes[:, 3] = predicted_bboxes[:, 3] + predicted_bboxes[:, 1]
    predicted_labels = torch.tensor([ele[1] for ele in predicted_res])

    gt_res = from_image_id_to_gtbboxes[image_id]
    gt_bboxes = torch.tensor([ele[0] for ele in gt_res])
    # convert from xywh to xyxy
    gt_bboxes[:, 2] = gt_bboxes[:, 2] + gt_bboxes[:, 0]
    gt_bboxes[:, 3] = gt_bboxes[:, 3] + gt_bboxes[:, 1]    
    gt_labels = torch.tensor([ele[1] for ele in gt_res])
    real_iou = iou_calculator(predicted_bboxes, gt_bboxes)
    #print(predicted_bboxes.shape, gt_bboxes.shape, real_iou.shape)
    #break
    max_iou_per_pred_val, max_iou_per_pred_idx = torch.max(real_iou, dim=-1)
    # convert the bbox from xyxy back to xywh
    predicted_bboxes[:, 2] = predicted_bboxes[:, 2] - predicted_bboxes[:, 0]
    predicted_bboxes[:, 3] = predicted_bboxes[:, 3] - predicted_bboxes[:, 1]
    # handle all the bboxes on by one
    for pred_bbox, pred_label, max_iou_val, max_iou_idx in zip(predicted_bboxes, predicted_labels, max_iou_per_pred_val, max_iou_per_pred_idx):
        if max_iou_val < 0.5:
            # if the max iou < 0.5, just maintain the predicted label but change the confidence score to 0
            info = {'image_id': image_id, 'bbox': pred_bbox.tolist(), 'score': 0.0, 'category_id': pred_label.item()}
        else:
            # find the max iou idx and assign the gtlabel to the prediction
            assigned_label = gt_labels[max_iou_idx]
            info = {'image_id': image_id, 'bbox': pred_bbox.tolist(), 'score': 1.0, 'category_id': assigned_label.item()}
        all_final_prediction.append(info)

print(len(all_final_prediction))
file = open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/fixed_label_prediction.json', 'w')
file.write(json.dumps(all_final_prediction))
file.close()