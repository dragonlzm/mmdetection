import json
import random
import requests
import os
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from copy import deepcopy
import torch
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D


#proposal_file_path = "/data/zhuoming/detection/test/proposal_selection_v1/testing.gt_acc.json"
#proposal_result = json.load(open(proposal_file_path))
proposal_path_root = 'data/coco/clip_proposal/32_32_512'


#gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
gt_annotation_path = "/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json"
gt_anno_result = json.load(open(gt_annotation_path))


#from_img_id_to_pred = {}
from_img_id_to_gt = {}

# for ele in proposal_result:
#     img_id = ele['image_id']
#     bbox = ele['score']
#     from_img_id_to_pred[img_id] = bbox

for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append(bbox)

all_img_info = {info['id']:info for info in gt_anno_result['images']}

save_root = '/home/zhuoming/most_accurate_proposal/'
#save_root = '/home/zhuoming/results_16_16_1024_nms07_novel_/'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07_novel\\'

target_list = [558840, 200365, 495357, 116061, 16164, 205350, 74, 212545, 514915, 154589, 471175, 225919, 400728, 
               194306, 383780, 580255, 370210, 75283, 325969, 251716, 13882, 185156, 176697, 376608, 178939, 173350, 
               26654, 346071, 158497, 408307, 252203, 263146, 390348, 395230, 426342, 155997, 278435, 47263, 519838, 
               283119, 369190, 458424, 239985, 151988, 364010, 205573, 427639, 233660, 32054, 153692, 174871]

iou_calculator = BboxOverlaps2D()

for i, img_id in enumerate(from_img_id_to_gt):
    if img_id not in target_list:
        continue
    url = all_img_info[img_id]['coco_url']
    file_name = all_img_info[img_id]['file_name']
    save_path = os.path.join(save_root, file_name)
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    im = np.array(Image.open(save_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    
    # load the proposal
    pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    xyxy_proposal = torch.tensor(pregenerated_bbox['score'])[:,:4]

    if img_id in from_img_id_to_gt:
        for box in from_img_id_to_gt[img_id]:
            rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
            xyxy_gt = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]])
            # find the proposal 
            iou = iou_calculator(xyxy_gt, xyxy_proposal)
            # if the gt bbox does not overlap with anyone of the proposal
            # does not draw any proposal for this gt bbox
            max_value, max_id = torch.max(iou, dim=-1)
            if max_value == 0:
                continue
            target_proposal = xyxy_proposal[max_id.item()]
            # draw the proposal
            rect = patches.Rectangle((target_proposal[0], target_proposal[1]),
                                     target_proposal[2]-target_proposal[0],
                                     target_proposal[3]-target_proposal[1], linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    #for box in from_img_id_to_pred[img_id]:
        #box = annotation['bbox']
    #    rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
    #    ax.add_patch(rect)

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path, file_name))
    plt.close()