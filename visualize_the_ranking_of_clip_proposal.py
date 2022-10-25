from email.mime import base
import json
import torch 
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
import os
iou_calculator = BboxOverlaps2D()

novel_cate_id = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
base_cate_id = [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]


annotation_file = "data/coco/annotations/coco_10img_per_novel.json"
target_res_file_root = '/data/zhuoming/code/new_rpn/mmdetection/data/coco/experiment/imagenet1762/random/'
# get all gt bbox
annotation_info = json.load(open(annotation_file))
from_image_id_to_bbox = {}
save_root = '/home/zhuoming/rank_visualization/'

for anno in annotation_info['annotations']:
    image_id = anno['image_id']
    bbox = anno['bbox']
    category_id = anno['category_id']
    bbox.append(category_id)
    if image_id not in from_image_id_to_bbox:
        from_image_id_to_bbox[image_id] = []
    from_image_id_to_bbox[image_id].append(bbox)

novel_list_obj_rank = []
novel_list_base_rank = []

base_list_obj_rank = []
base_list_base_rank = []

for image_id in from_image_id_to_bbox:
    all_gt_bboxes = torch.tensor(from_image_id_to_bbox[image_id])
    xyxy_gt = all_gt_bboxes[:, :4]
    xyxy_gt[:, 2] = xyxy_gt[:, 0] + xyxy_gt[:, 2]
    xyxy_gt[:, 3] = xyxy_gt[:, 1] + xyxy_gt[:, 3]

    # get all clip proposals
    json_file_name = str(image_id).zfill(12) + '.json'
    clip_proposals = torch.tensor(json.load(open(os.path.join(target_res_file_root, json_file_name)))['res'])
    clip_proposal_bbox = clip_proposals[:, :4]

    # calculate the iou between all clip proposal and all gt bbox
    real_iou = iou_calculator(clip_proposal_bbox, xyxy_gt)
    max_iou_val, max_idx = torch.max(real_iou, dim=-1)

    # for each clip proposal to see which gt bbox has highest overlap, and get the categories id and judge whether it's a novel bbox
    meaningful_clip_proposal_idx = (max_iou_val != 0)
    meaningful_clip_proposal_matched_gt_idx = max_idx[meaningful_clip_proposal_idx]
    meaningful_clip_proposal_matched_gt_info = all_gt_bboxes[meaningful_clip_proposal_matched_gt_idx]
    meaningful_clip_proposal_clip_info = clip_proposals[meaningful_clip_proposal_idx]

    # assign color to each clip proposal base on the categories
    for clip_info, gt_info in zip(meaningful_clip_proposal_clip_info, meaningful_clip_proposal_matched_gt_info):
        obj_rank, base_rank = clip_info[-2].item(), clip_info[-1].item()
        gt_cate = gt_info[-1].item()
        if gt_cate in novel_cate_id:
            novel_list_obj_rank.append(obj_rank)
            novel_list_base_rank.append(base_rank)
        elif gt_cate in base_cate_id:
            base_list_obj_rank.append(obj_rank)
            base_list_base_rank.append(base_rank)
        else:
            novel_list_obj_rank.append(obj_rank)
            novel_list_base_rank.append(base_rank)

import matplotlib.pyplot as plt
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.scatter(novel_list_obj_rank, novel_list_base_rank, s=0.1, c='red', alpha=0.5)
plt.scatter(base_list_obj_rank, base_list_base_rank, s=0.1, c='green', alpha=0.5)

#plt.title('distribution of base prediction, conf vs iou')
#file_name = 'obj_rank_vs_base_rank_' + str(image_id) + '.png'
file_name = 'obj_rank_vs_base_rank_all_red_first.png'
#file_name = 'obj_rank_vs_base_rank_all_green_first.png'
plt.title(file_name)
#plt.title('distribution of valid prediction, conf vs max(iop, iog)')
#plt.title('distribution of invalid prediction')
#plt.xlabel('max(iop, iog)')
plt.xlabel('obj_rank')

plt.ylabel('base_rank')
#plt.show()

plt.title('obj_rank_vs_base_rank')
plt.savefig(os.path.join(save_root, file_name))
plt.close()