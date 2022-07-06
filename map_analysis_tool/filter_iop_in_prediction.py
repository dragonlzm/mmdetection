# this script aims to filter the prediction with iop with any base gt
# iop > 0.7 and iou with any other gt bboxes lower than 0.5
# if it's the same gt bboxes than it should be filtered
# otherwise maintrain such bboxes


import json
import torch
import sys
import os
sys.path.append('../mmdetection')
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D

iou_calculator = BboxOverlaps2D()
# aggregate the prediction of the model base on the image and base on the categories
# aggregate the gt on each image base on the categoires
gt_path = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'
gt_anno = json.load(open(gt_path))

from_image_id_to_gtbboxes = {}
# collect the annotations for both base and novel
for anno in gt_anno['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    if image_id not in from_image_id_to_gtbboxes:
        from_image_id_to_gtbboxes[image_id] = {}
    if category_id not in from_image_id_to_gtbboxes[image_id]:
        from_image_id_to_gtbboxes[image_id][category_id] = []
    from_image_id_to_gtbboxes[image_id][category_id].append(bbox)

pred_paths = ['/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/base_results.bbox.json', 
                '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json']

for pred_path in pred_paths:
    # we aggregate the prediction on each image base on the categories
    pred_res = json.load(open(pred_path))
    from_image_id_to_prediction = {}
    for pred in pred_res:
        image_id = pred['image_id']
        bbox = pred['bbox']
        category_id = pred['category_id']
        score = pred['score']
        bbox.append(score)
        if image_id not in from_image_id_to_prediction:
            from_image_id_to_prediction[image_id] = {}
        if category_id not in from_image_id_to_prediction[image_id]:
            from_image_id_to_prediction[image_id][category_id] = []
        from_image_id_to_prediction[image_id][category_id].append(bbox)

    # for each image calculate the iou between the gt and prediction
    all_final_prediction = []

    for image_id in from_image_id_to_prediction:
        pred_on_this_image = from_image_id_to_prediction[image_id]
        # if image does not have gt
        if image_id not in from_image_id_to_gtbboxes:
            # do something to let the prediction back to the result
            for cate_id in pred_on_this_image:
                for pred_bbox in pred_on_this_image[cate_id]:
                    info = {'image_id': image_id, 'bbox': pred_bbox[:-1], 'score': pred_bbox[-1], 'category_id': cate_id}
                    all_final_prediction.append(info)
            continue
        # we do not need to handle the situation where there is gt but no prediction
        gt_on_this_image = from_image_id_to_gtbboxes[image_id]
        for cate_id in pred_on_this_image:
            # if there is prediction but no gt of this cate 
            # set the confidence score to 0
            if cate_id not in gt_on_this_image:
                for pred_bbox in pred_on_this_image[cate_id]:
                    info = {'image_id': image_id, 'bbox': pred_bbox[:-1], 'score': pred_bbox[-1], 'category_id': cate_id}
                    all_final_prediction.append(info)
                continue
            # for the situation that there are prediction and gt at the same time 
            all_gt_bboxes = torch.tensor(gt_on_this_image[cate_id])
            all_pred_bboxes = torch.tensor(pred_on_this_image[cate_id])
            # convert xywh to xyxy
            all_gt_bboxes[:, 2] = all_gt_bboxes[:, 0] + all_gt_bboxes[:, 2]
            all_gt_bboxes[:, 3] = all_gt_bboxes[:, 1] + all_gt_bboxes[:, 3]
            all_pred_bboxes[:, 2] = all_pred_bboxes[:, 0] + all_pred_bboxes[:, 2]
            all_pred_bboxes[:, 3] = all_pred_bboxes[:, 1] + all_pred_bboxes[:, 3]        
            # real_iou = iou_calculator(all_pred_bboxes, all_gt_bboxes)
            # max_iou_per_pred_val, max_iou_per_pred_idx = torch.max(real_iou, dim=-1)
            iop = iou_calculator(all_pred_bboxes, all_gt_bboxes, mode='iof')
            max_iop_per_pred, max_iop_per_pred_idx = torch.max(iop, dim=-1)
            
            iou = iou_calculator(all_pred_bboxes, all_gt_bboxes)
            #print('iou shape', iou.shape)
            max_iou_per_pred, max_iou_per_pred_idx = torch.max(iou, dim=-1)
            
            # convert xyxy back to xywh
            all_pred_bboxes[:, 2] = all_pred_bboxes[:, 2] - all_pred_bboxes[:, 0]
            all_pred_bboxes[:, 3] = all_pred_bboxes[:, 3] - all_pred_bboxes[:, 1]
            #for pred_bbox, max_iop_val, max_iop_idx, iou_per_pred in zip(all_pred_bboxes, max_iop_per_pred, max_iop_per_pred_idx, iou):
            for pred_bbox, max_iop_val, max_iou_val in zip(all_pred_bboxes, max_iop_per_pred, max_iou_per_pred):
                if max_iop_val > 0.9 and max_iou_val < 0.5:
                #if max_iop_val > 0.9 and iou_per_pred[max_iop_idx] < 0.5:
                    info = {'image_id': image_id, 'bbox': pred_bbox[:-1].tolist(), 'score': 0.0, 'category_id': cate_id}
                else:
                    info = {'image_id': image_id, 'bbox': pred_bbox[:-1].tolist(), 'score': pred_bbox[-1].item(), 'category_id': cate_id}
                all_final_prediction.append(info)

    print(len(all_final_prediction))
    save_path = os.path.join('/'.join(pred_path.split('/')[:-1]), 'filter_iop_prediction.json')
    file = open(save_path, 'w')
    file.write(json.dumps(all_final_prediction))
    #print(all_final_prediction[0])
    file.close()