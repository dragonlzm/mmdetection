# this script aims to evaluate the prediction from the perspective of the predition
# in other word, we judge each prediction(the bbox and the categories is meaningful)
# use three different criteria: iou > 0.5, iog > 0.5 and iop > 0.5
# if the prediction meet one the above criteria, then the prediction is regarded as meaningful
# aggregate the number of meaningful bboxes compare with each other.
import json
import torch
import sys
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

# we aggregate the prediction on each image base on the categories
pred_paths = ['/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/base_results.bbox.json', 
              '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48/base_results.bbox.json',
              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json',
              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results_e18.bbox.json',
              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results.bbox.json']
#pred_paths = ['/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_results_e18.bbox.json',
#              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_results.bbox.json']

#pred_path = '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/base_results.bbox.json'
#pred_path = '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48/base_results.bbox.json'
#pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json'
#pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results_e18.bbox.json'
#pred_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results.bbox.json'
iou_thresholds = [0.5, 0.7, 0.9]
#iou_thresholds = [0.9]

#iou_threshold = 0.5
#iou_threshold = 0.7
#iou_threshold = 0.9
#methods = ['iog', 'iop', 'iog_or_iop', 'iou']
#method = 'iog'
#method = 'iop'
#method = 'iog_or_iop'
#method = 'iou'
methods = ['iop_and_iou']

for pred_path in pred_paths:
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

    # for each image calculate the iou between the gt and prediction
    for method in methods:
        for iou_threshold in iou_thresholds:
            print('pred_path', pred_path, 'method', method, 'iou_threshold', iou_threshold)
            total_prediction = 0
            valid_prediction = 0
            for image_id in from_image_id_to_prediction:
                pred_on_this_image = from_image_id_to_prediction[image_id]
                #if len(pred_on_this_image.keys()) > 48:
                #    print('cates > 48', len(pred_on_this_image.keys()))
                # if image does not have gt
                if image_id not in from_image_id_to_gtbboxes:
                    # if the image does not have any gt bboxes
                    # we should not consider this image since all the prediction is false positive
                    continue
                
                # we do not need to handle the situation where there is gt but no prediction
                gt_on_this_image = from_image_id_to_gtbboxes[image_id]
                for cate_id in pred_on_this_image:
                    total_prediction += len(pred_on_this_image[cate_id])
                    #1. if there are predictions but no gt of this cate
                    # then all the predictions are false postive
                    if cate_id not in gt_on_this_image:
                        continue
                    
                    #2. for the situation that there are prediction and gt at the same time 
                    all_gt_bboxes = torch.tensor(gt_on_this_image[cate_id])
                    all_pred_bboxes = torch.tensor(pred_on_this_image[cate_id])
                    # convert xywh to xyxy
                    all_gt_bboxes[:, 2] = all_gt_bboxes[:, 0] + all_gt_bboxes[:, 2]
                    all_gt_bboxes[:, 3] = all_gt_bboxes[:, 1] + all_gt_bboxes[:, 3]
                    all_pred_bboxes[:, 2] = all_pred_bboxes[:, 0] + all_pred_bboxes[:, 2]
                    all_pred_bboxes[:, 3] = all_pred_bboxes[:, 1] + all_pred_bboxes[:, 3]
                    if 'iog' in method:
                        # calculate the iog
                        iog = iou_calculator(all_gt_bboxes, all_pred_bboxes, mode='iof')
                        # need to transpose the iou
                        iog = iog.permute([1,0])
                        if iog.shape[-1] >= 2:
                            top2_iog_per_pred, top2_iog_per_pred_idx = torch.topk(iog, 2, dim=-1)
                            max_iog_per_pred = top2_iog_per_pred[:, 0]
                            second_max_iog_per_pred = top2_iog_per_pred[:, 1]
                            iog_valid_idx = ((max_iog_per_pred > iou_threshold) & (second_max_iog_per_pred < 0.5))
                        else:
                            max_iog_per_pred, max_iog_per_pred_idx = torch.max(iog, dim=-1)
                            iog_valid_idx = (max_iog_per_pred > iou_threshold)
                    
                    # calculate the iop
                    if 'iop' in method:
                        iop = iou_calculator(all_pred_bboxes, all_gt_bboxes, mode='iof')
                        #print('iop shape', iop.shape)
                        max_iop_per_pred, max_iop_per_pred_idx = torch.max(iop, dim=-1)
                        #print('max_iop_per_pred_idx shape', max_iop_per_pred_idx.shape)
                        
                    # calculate the iou
                    if 'iou' in method:
                        iou = iou_calculator(all_pred_bboxes, all_gt_bboxes)
                        #print('iou shape', iou.shape)
                        max_iou_per_pred, max_iou_per_pred_idx = torch.max(iou, dim=-1)
                    
                    # the valid idx 
                    if method == 'iog':
                        valid_idx = iog_valid_idx
                    elif method == 'iop':
                        valid_idx = (max_iop_per_pred > iou_threshold)
                    elif method == 'iou':
                        valid_idx = (max_iou_per_pred > iou_threshold)
                    elif method == 'iop_and_iou':
                        max_iop_per_pred_idx = max_iop_per_pred_idx.unsqueeze(dim=-1)
                        # repective_iou = torch.take_along_dim(iou, max_iop_per_pred_idx,dim=-1).squeeze(dim=-1)
                        # valid_idx = ((max_iop_per_pred > iou_threshold) & (repective_iou < 0.5))
                        
                        valid_idx = ((max_iop_per_pred > iou_threshold) & (max_iou_per_pred < 0.5))
                    else:
                        valid_idx = (iog_valid_idx | (max_iop_per_pred > iou_threshold))
                    valid_num = torch.sum(valid_idx)
                    valid_prediction += valid_num

            print('total_prediction', total_prediction, 'valid_prediction', valid_prediction,
                'valid percent:', (valid_prediction/total_prediction)*100)
