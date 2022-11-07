# this script aims to visualize the lvis dataset, include the filtering with base

from email.mime import base
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
gt_content = json.load(open('/data/zhuoming/detection/lvis_v1/annotations/lvis_train_novel_base_subset.json'))

novel_id = []
base_id = []
for ele in gt_content['categories']:
    frequency = ele['frequency']
    category_id = ele['id']
    if frequency == 'c' or frequency == 'f':
        base_id.append(category_id)
    else:
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
save_root = '/home/zhuoming/lvis_visualization_most_matched'
proposal_path_root = '/data/zhuoming/detection/lvis_v1/clip_proposal_feat/lvis_base_finetuned/random'

for i, image_id in enumerate(from_image_id_to_annotation):
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
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    # print the base
    for box in from_image_id_to_annotation[image_id]['base']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    # print the novel
    for box in from_image_id_to_annotation[image_id]['novel']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # load the proposal 
    pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    all_proposals = torch.tensor(pregenerated_bbox['bbox'])
    ## scale the bbox back to the original size
    img_metas = pregenerated_bbox['img_metas']
    pre_extract_scale_factor = np.array(img_metas['scale_factor']+[1,1]).astype(np.float32)
    # all_proposals: [562.7451782226562, 133.49032592773438, 653.2548217773438, 314.5096740722656, 0.9763965010643005, 461.0]
    all_proposals = all_proposals / pre_extract_scale_factor
    
    # find the matched bbox for gt bbox "novel"
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])
    if novel_gt_bboxes.shape[0] != 0:
        match = False
        all_predicted_cate = all_proposals[:, -1]
        for novel_bbox in novel_gt_bboxes:
            novel_bbox_cate_id = novel_bbox[-1]
            #print('novel_bbox', novel_bbox)
            xyxy_gt = torch.tensor([[novel_bbox[0], novel_bbox[1], novel_bbox[0] + novel_bbox[2], 
                                novel_bbox[1] + novel_bbox[3]]])
            #print('xyxy_gt', xyxy_gt)
            real_iou = iou_calculator(xyxy_gt, all_proposals[:, :4])
            iou_over_05 = (real_iou >= 0.1)
            remain_proposal = all_proposals[iou_over_05.squeeze(dim=0)]
            for box in remain_proposal:
                cate = box[-1]
                if cate == novel_bbox_cate_id:
                    color = 'r'
                else:
                    color = 'yellow'
                rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor=color,facecolor='none')
                ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()

