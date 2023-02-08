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

def get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
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
    left = torch.clamp(left, min=-0.5, max=0)
    right = torch.clamp(right, min=-0.5, max=0)
    top = torch.clamp(top, min=-0.5, max=0)
    bottom = torch.clamp(bottom, min=-0.5, max=0)
    # the result is a tensor with size (pixel number, proposal number)
    # if the result == 0 mean the pixel is inside the bboxes
    result = left + right + top + bottom
    
    return result

def assign_gt_label(assignment_result, predicted_labels):
    assigned_label = []
    for score_per_proposal, proposal_idx in zip(assignment_result, predicted_labels):
        temp = torch.full(score_per_proposal.shape, -1).long().cuda()
        temp[score_per_proposal == 0.0] = proposal_idx
        assigned_label.append(temp.unsqueeze(dim=0))
    assigned_label = torch.cat(assigned_label, dim=0)
    assigned_label = assigned_label.permute([1,0])
    #print(torch.sum(assigned_label != -1))
    # torch.Size([273280, 1000])
    #print(assigned_label.shape)
    most_propular_per_pixel = torch.full((assigned_label.shape[0],), 65).long().cuda()
    # use the stupid method to aggregate the result
    for i in range(assigned_label.shape[0]):
        if torch.sum(assigned_label[i]) == -1000:
            continue
        most_propular_per_pixel[i] = torch.mode(assigned_label[i][assigned_label[i] != -1])[0]
    return most_propular_per_pixel

proposal_path = "/home/zhuoming/detectron_proposal2"
save_path = "/home/zhuoming/label_assignment/"

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
    _, predicted_labels = torch.max(clip_score, dim=-1)
    # obtain the size of the image
    image_info = from_image_id_to_image_info[image_id]
    h, w = image_info['height'], image_info['width']
    # create the pixel map for each image
    all_points = get_points_single((h,w), torch.float16, torch.device('cuda'))
    # assign the result
    assignment_result = get_target_single(predict_boxes, predicted_labels, all_points)
    assignment_result = assignment_result.permute([1,0])
    
    # assign label
    assigned_label = assign_gt_label(assignment_result, predicted_labels)
    #print(torch.mode(assigned_label[i][assigned_label[i]!=-1])[0])
    #print(torch.sum(most_propular_per_pixel!=-1))
    # visualization_result = []
    # for pixel_val in assigned_label:
    #     visualization_result.append(color[pixel_val].unsqueeze(dim=0))
    
    # visualization_result = torch.cat(visualization_result, dim=0)
    visualization_result = color[assigned_label]
    visualization_result = visualization_result.reshape(h,w,3)
    #print('visualization_result', visualization_result.shape)
    
    # most_propular_per_pixel[most_propular_per_pixel == -1] = 0
    # #most_propular_per_pixel = most_propular_per_pixel + 100
    # most_propular_per_pixel *= (255 // 65)
    # # reshape the result back to the image
    # most_propular_per_pixel = most_propular_per_pixel.reshape(h,w)
    
    # # use the label as the value as the pixel color
    # most_propular_per_pixel = most_propular_per_pixel.unsqueeze(dim=0)
    # #print('most_propular_per_pixel', most_propular_per_pixel.shape)
    # most_propular_per_pixel = most_propular_per_pixel.repeat(3, 1, 1)
    # most_propular_per_pixel = most_propular_per_pixel.permute(1, 2, 0)
    # print('most_propular_per_pixel', torch.sum(most_propular_per_pixel!=-1))
    
    # # most_propular_per_pixel is the result
    # most_propular_per_pixel = torch.full((assigned_label.shape[0],), -1).long().cuda()
    # most_propular_per_pixel[torch.sum(assigned_label, dim=-1) != -1000] = torch.mode(assigned_label[torch.sum(assigned_label, dim=-1) != -1000], dim=-1)[0]
    # print(torch.sum(most_propular_per_pixel!=-1))
    # most_propular_per_pixel = torch.mode(assigned_label, dim=-1)[0]
    # #result torch.Size([273280, 1000]) most_propular_per_pixel torch.Size([273280])
    # #print('result', assigned_label, 'most_propular_per_pixel', most_propular_per_pixel)

    # visualize the color and draw the picture
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = Image.fromarray(np.uint8(visualization_result.cpu()))
    data.save(os.path.join(save_path, file_name + "_label_assign.png"))
    #break
