# this version draw a img for each novel categories bboxes
# this script aims to visualize the coco dataset, include the filtering with base

import json
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
iou_calculator = BboxOverlaps2D()

# load the gt bboxes
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json'))
#gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017.json'))
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))


all_names = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')
base_names = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_names = ('airplane', 'bus', 'cat', 'dog', 'cow', 
            'elephant', 'umbrella', 'tie', 'snowboard', 
            'skateboard', 'cup', 'knife', 'cake', 'couch', 
            'keyboard', 'sink', 'scissors')

from_name_to_idx = {name:i for i, name in enumerate(all_names)}
from_idx_to_name = {i:name for i, name in enumerate(all_names)}

from_gt_id_to_name = {}
novel_id = []
base_id = []
for ele in gt_content['categories']:
    category_id = ele['id']
    name = ele['name']
    from_gt_id_to_name[category_id] = name
    if name in base_names:
        base_id.append(category_id)
    elif name in novel_names:
        novel_id.append(category_id)

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    bbox.append(cate_id)
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    if cate_id in base_id:
        from_image_id_to_annotation[image_id]['base'].append(bbox)
    else:
        from_image_id_to_annotation[image_id]['novel'].append(bbox)

# collect the image info
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info

# load the proposal and print the image
#save_root = '/home/zhuoming/coco_visualization_most_matched'
save_root = '/home/zhuoming/coco_visualization_vit_most_matched_per_novel'
#proposal_path_root = '/data/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/random'
#proposal_path_root = '/home/zhuoming/detectron_proposal1'
proposal_path_root = '/home/zhuoming/detectron_proposal2'

count = 0
for i, image_id in enumerate(from_image_id_to_annotation):
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
    if novel_gt_bboxes.shape[0] == 0:
        continue
    else:
        count += 1
    if count > 20:
        break

    #download the image
    url = from_image_id_to_image_info[image_id]['coco_url']
    file_name = from_image_id_to_image_info[image_id]['coco_url'].split('/')[-1]
    save_path = os.path.join(save_root, file_name)
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    im = np.array(Image.open(save_path), dtype=np.uint8)
    if novel_gt_bboxes.shape[0] != 0:
        for novel_box in from_image_id_to_annotation[image_id]['novel']:
            # show the image
            fig,ax = plt.subplots(1)
            ax.imshow(im)
            
            rect = patches.Rectangle((novel_box[0], novel_box[1]),novel_box[2],novel_box[3],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
        
            # load the proposal 
            #pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
            pregenerate_prop_path = os.path.join(proposal_path_root, (file_name + '_final_pred.json'))
            pregenerated_bbox = json.load(open(pregenerate_prop_path))
            #all_proposals = torch.tensor(pregenerated_bbox['bbox'])
            all_proposals = torch.tensor(pregenerated_bbox['box'])
            
            # load the clip score
            clip_path = os.path.join(proposal_path_root, (file_name + '_clip_pred.json'))
            clip_content = json.load(open(clip_path))        
            all_clip_score = torch.tensor(clip_content['score'])
            
            # match the novel bboxes with the 
            match = False
            all_predicted_cate = all_proposals[:, -1]
            novel_bbox_cate_id = novel_box[-1]
            #print('novel_bbox', novel_bbox)
            xyxy_gt = torch.tensor([[novel_box[0], novel_box[1], novel_box[0] + novel_box[2], 
                                novel_box[1] + novel_box[3]]])
            #print('xyxy_gt', xyxy_gt)
            real_iou = iou_calculator(xyxy_gt, all_proposals[:, :4])
            iou_idx_over_zero = (real_iou > 0)
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            else:
                value, idx = torch.max(real_iou, dim=-1)
                if value < 0.5:
                    continue
                remain_proposal = all_proposals[idx].squeeze(dim=0)
                rect = patches.Rectangle((remain_proposal[0], remain_proposal[1]),remain_proposal[2]-remain_proposal[0],remain_proposal[3]-remain_proposal[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                
                # find the predicted categories
                #print('all_clip_score', all_clip_score.shape, idx)
                clip_score_for_proposal = all_clip_score[idx]
                clip_score_for_proposal = clip_score_for_proposal.squeeze(dim=0)
                #print('clip_score_for_proposal', clip_score_for_proposal.shape)
                max_clip_score_val, max_clip_score_idx = torch.max(clip_score_for_proposal, dim=-1)
                pred_cate_name = from_idx_to_name[max_clip_score_idx.item()]
                novel_cate_name = from_gt_id_to_name[novel_bbox_cate_id]
                #print('clip_score_for_proposal', clip_score_for_proposal, 'max_clip_score_val', max_clip_score_val , 'max_clip_score_idx', max_clip_score_idx, 'pred_cate_name', pred_cate_name, 'novel_cate_name', novel_cate_name)

                print_path = os.path.join(save_root, 'printed')
                if not os.path.exists(print_path):
                    os.makedirs(print_path)
                save_file_name = file_name + '_'+ novel_cate_name + '_' + pred_cate_name + '_' + '.jpg'
                plt.savefig(os.path.join(print_path, save_file_name))
                plt.close()
    else:
        continue

