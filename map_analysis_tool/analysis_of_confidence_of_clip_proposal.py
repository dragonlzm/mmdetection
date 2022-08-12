# this script is aim to calculate the average confidence of the matched predictions
import json
import torch
import sys
import os
import numpy as np
sys.path.append('../mmdetection')
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()
from ..script_utils import from_image_id_to_file_name


def aggregate_conf_score_of_matched_clip_proposal(gt_path, clip_proposal_root, save_file_prefix):
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
    all_valid_confidence_score = []
    invalid_prediction_num = 0
    invalid_prediction_conf = 0
    all_invalid_confidence_score = []

    for image_id in from_image_id_to_gt:
        # load the proposal result
        proposal_file_name = os.path.join(clip_proposal_root, (from_image_id_to_file_name(image_id) + '.json'))
        proposal_file_content = json.load(proposal_file_name)
        # the format of the rand_bbox is xyxy
        rand_bbox = np.array(proposal_file_content['bbox']).astype(np.float32)
        
        # reshape the proposal back to normal scale
        pre_extract_scale_factor = np.array(proposal_file_content['img_metas']['scale_factor']).astype(np.float32)
        final_rand_bbox = rand_bbox / pre_extract_scale_factor
        final_rand_bbox = final_rand_bbox.tolist()
        
        # aggregate the clip proposal base on the categories
        now_image_clip_proposal = {}
        for bbox in final_rand_bbox:
            category_id = bbox.pop(-1)
            if category_id not in final_rand_bbox:
                now_image_clip_proposal[category_id] = []
            now_image_clip_proposal[category_id].append(bbox)
        
        for category_id in from_image_id_to_gt[image_id]:
            pred_for_this_cate = torch.tensor(now_image_clip_proposal[category_id])
            gt_for_this_cate = torch.tensor(from_image_id_to_gt[image_id][category_id])
            # convert_from_xywh_to_xyxy
            #pred_for_this_cate[:,2] = pred_for_this_cate[:,2] + pred_for_this_cate[:,0]
            #pred_for_this_cate[:,3] = pred_for_this_cate[:,3] + pred_for_this_cate[:,1]
            gt_for_this_cate[:,2] = gt_for_this_cate[:,2] + gt_for_this_cate[:,0]
            gt_for_this_cate[:,3] = gt_for_this_cate[:,3] + gt_for_this_cate[:,1]
            
            iou = iou_calculator(pred_for_this_cate, gt_for_this_cate)
            #print('iou shape', iou.shape)
            max_iou_per_pred, max_iou_per_pred_idx = torch.max(iou, dim=-1)
            
            valid_prediction = pred_for_this_cate[max_iou_per_pred >= 0.5]
            invalid_prediction = pred_for_this_cate[max_iou_per_pred < 0.5] 
            if valid_prediction.shape[0] > 0:
                valid_prediction_num += valid_prediction.shape[0]
                valid_prediction_conf += torch.sum(valid_prediction[:, -1]).item()
                all_valid_confidence_score.append(valid_prediction[:, -1])
            if invalid_prediction.shape[0] > 0:
                invalid_prediction_num += invalid_prediction.shape[0]
                invalid_prediction_conf += torch.sum(invalid_prediction[:, -1]).item()
                all_invalid_confidence_score.append(invalid_prediction[:, -1])

    print(valid_prediction_num, valid_prediction_conf, valid_prediction_conf / valid_prediction_num)
    print(invalid_prediction_num, invalid_prediction_conf, invalid_prediction_conf / invalid_prediction_num)
    all_invalid_confidence_score = torch.cat(all_invalid_confidence_score, dim=0)
    all_valid_confidence_score = torch.cat(all_valid_confidence_score, dim=0)

    torch.save(all_valid_confidence_score, save_file_prefix + '_all_valid_confidence_score.pt')
    torch.save(all_invalid_confidence_score, save_file_prefix + '_all_invalid_confidence_score.pt')

if __name__ == "__main__":
    gt_path = gt_path = '/data/zhuoming/detection/coco/annotations/instances_train2017_0_8000_novel17.json'
    proposal_path_root = ''
    aggregate_conf_score_of_matched_clip_proposal(gt_path, proposal_path_root)