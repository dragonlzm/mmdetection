# this script aims to vote the pixels on the feature map of each image using the clip proposal
import json
import torch
from torch import nn
import requests
import os
import numpy as np
from PIL import Image
INF = 1e8

def get_points_single(featmap_size,
                        dtype,
                        device):
    """Get points of a single scale level."""
    h, w = featmap_size
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    x_range = torch.arange(w, device=device).to(dtype)
    y_range = torch.arange(h, device=device).to(dtype)
    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        
    return points

def get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    """Compute regression and classification targets for a single image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), num_classes), \
                gt_bboxes.new_zeros((num_points, 4))
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
        gt_bboxes[:, 3] - gt_bboxes[:, 1])
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)
    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    # condition2: limit the regression range for each location
    max_regress_distance = bbox_targets.max(-1)[0]
    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    areas[inside_gt_bbox_mask == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)
    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = num_classes  # set as BG
    bbox_targets = bbox_targets[range(num_points), min_area_inds]
    return labels, bbox_targets

proposal_path = "/home/zhuoming/detectron_proposal2"

# load the gt infomation for each image
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

# for each image
for image_id in from_image_id_to_image_info:
    file_name = from_image_id_to_image_info[image_id]['file_name']
    # load the vitdet final prediction 
    prediction_file_name = file_name + '_final_pred.json'
    prediction_path = os.path.join(proposal_path, prediction_file_name)
    prediction_file_content = json.load(open(prediction_path))
    predict_boxes = torch.tensor(prediction_file_content['box']).cuda()
    # load the clip confidence score
    clip_conf_file_name = file_name + "_clip_pred.json"
    clip_conf_path = os.path.join(proposal_path, clip_conf_file_name)
    clip_conf_file_content = json.load(open(clip_conf_path))
    clip_score = torch.tensor(clip_conf_file_content['score']).cuda()
    _, predicted_labels = torch.max(clip_score)
    # obtain the size of the image
    image_info = from_image_id_to_image_info[image_id]
    h, w = image_info['height'], image_info['width']
    # create the pixel map for each image
    all_points = get_points_single((h,w), torch.float,  torch.device('cuda'))
    # assign the result
    result = get_target_single(predict_boxes, predicted_labels, all_points, regress_ranges)
    print('result', result.shape)
    break

# load the clip predict score and the vitdet final prediction for each image

# expect for each pixel we store the 
