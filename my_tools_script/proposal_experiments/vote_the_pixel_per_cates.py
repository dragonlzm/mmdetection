# this script aims to vote the pixels on the feature map of each image using the clip proposal
import json
import torch
from torch import nn
import requests
import os
import numpy as np
from PIL import Image
INF = 1e8
torch.manual_seed(0)
color = torch.randint(0, 255, (66, 3))
color[65] = torch.tensor([0,0,0]) 

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

def _get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    """Compute regression and classification targets for a single image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), num_classes), \
                gt_bboxes.new_zeros((num_points, 4))
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
    return inside_gt_bbox_mask


def get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    # this function return the per categories mask
    num_points = points.size(0)
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
    gt_bboxes[:, 3] - gt_bboxes[:, 1])
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    
    total_iteration = 10
    gap_per_iter = predict_boxes.shape[0] / total_iteration
    all_result = []
    for i in range(total_iteration):
        start = int(i * gap_per_iter)
        end = int((i+1) * gap_per_iter)
        #print('start:end', start, end)
        _inside_gt_bbox_mask = _get_target_single(gt_bboxes[start:end], gt_labels[start:end], all_points)
        #print(temp_assigned_label.shape)
        all_result.append(_inside_gt_bbox_mask)
    # inside_gt_bbox_mask will be a tensor([num_of_pixel, num_of_proposal]), a true/ false mask
    inside_gt_bbox_mask = torch.cat(all_result, dim=-1)
    print(inside_gt_bbox_mask)
    
    areas[inside_gt_bbox_mask == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)
    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = num_classes  # set as BG
    return labels    


proposal_path = "/home/zhuoming/detectron_proposal2"
save_path = "/home/zhuoming/label_assignment_min_area/"

# load the gt infomation for each image
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

# for each image
for i, image_id in enumerate(from_image_id_to_image_info):
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
    _, predicted_labels = torch.max(clip_score, dim=-1)
    # obtain the size of the image
    image_info = from_image_id_to_image_info[image_id]
    h, w = image_info['height'], image_info['width']
    # create the pixel map for each image
    all_points = get_points_single((h,w), torch.float16, torch.device('cuda'))
    # assign the result
    assigned_label = get_target_single(predict_boxes, predicted_labels, all_points)
    break
    
    # visualization_result = torch.cat(visualization_result, dim=0)
    visualization_result = color[assigned_label]
    visualization_result = visualization_result.reshape(h,w,3)
    #print('visualization_result', visualization_result.shape)

    # visualize the color and draw the picture
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = Image.fromarray(np.uint8(visualization_result.cpu()))
    data.save(os.path.join(save_path, file_name + "_label_assign.png"))
    if i > 10:
        break
