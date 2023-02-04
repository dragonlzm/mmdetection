# this script aims to visualize the prediction from the one-stage detector and the two-stage detector
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

path_for_maskrcnn = "/data/zhuoming/detection/grad_clip_check/mask_rcnn_distill_301212_3xschedule/base_and_novel.bbox.json"
path_for_fcos = "/data/zhuoming/detection/one_stage/fcos_distill_wcenterness_fixnorm_2xreg_2xdist_3xschedule/base_and_novel.bbox.json"

predict_path = path_for_maskrcnn


# load and aggregate the predictions
pred_content = json.load(open(predict_path))
# aggregate the predition base on the image
from_image_id_to_prediction = {}
for res in pred_content:
    image_id = res['image_id']
    bbox = res['bbox']
    score = res['score']
    category_id = res['category_id']
    # the bbox from the prediction will be [x,y,w,h,score,cate_id]
    bbox.append(score)
    bbox.append(category_id)
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = {'bboxes':[]}
    from_image_id_to_prediction[image_id]['bboxes'].append(bbox)



# load the gt and aggregate the gt
gt_anno_file = '/data/zhuoming/detection/coco/annotations/instances_val2017.json'

from_image_id_to_image_file_name = {}

# aggregate the gt bboxes base on the image
gt_content = json.load(open(gt_anno_file))

base_cates_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

from_cate_name_to_cate_id = {anno['name']: anno['id'] for anno in gt_content['categories']}
from_cate_id_to_cate_name = {anno['id']: anno['name'] for anno in gt_content['categories']}
base_cates_ids = [from_cate_name_to_cate_id[name] for name in base_cates_name]

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    bbox.append(cate_id)
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    if cate_id in base_cates_ids:
        from_image_id_to_annotation[image_id]['base'].append(bbox)
    else:
        from_image_id_to_annotation[image_id]['novel'].append(bbox)
        
# collect the image info
from_image_id_to_image_info = {}
for info in gt_content['images']:
    image_id = info['id']
    from_image_id_to_image_info[image_id] = info
    
    
count = 0
#save_root = '/home/zhuoming/test'
#save_root = '/home/zhuoming/raw_rpn'

#save_root = '/home/zhuoming/retina'
#save_root = '/home/zhuoming/mrcnn'
#save_root = '/home/zhuoming/fcos_3x'
save_root = '/home/zhuoming/mrcnn_3x'

for i, image_id in enumerate(from_image_id_to_annotation):
    # print the prediction
    all_prediction = torch.tensor(from_image_id_to_prediction[image_id]['bboxes'])
    all_prediction[:, 2] = all_prediction[:, 0] + all_prediction[:, 2]
    all_prediction[:, 3] = all_prediction[:, 1] + all_prediction[:, 3]
    base_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['base'])
    novel_gt_bboxes = torch.tensor(from_image_id_to_annotation[image_id]['novel'])

    # we only visualize the image with novel instance
    if novel_gt_bboxes.shape[0] == 0:
        continue
    else:
        count += 1
    if count > 20:
        break
    
    if base_gt_bboxes.shape[0] != 0:
        base_gt_bboxes[:, 2] = base_gt_bboxes[:, 0] + base_gt_bboxes[:, 2]
        base_gt_bboxes[:, 3] = base_gt_bboxes[:, 1] + base_gt_bboxes[:, 3]    
    if novel_gt_bboxes.shape[0] != 0:
        novel_gt_bboxes[:, 2] = novel_gt_bboxes[:, 0] + novel_gt_bboxes[:, 2]
        novel_gt_bboxes[:, 3] = novel_gt_bboxes[:, 1] + novel_gt_bboxes[:, 3]
        
    # download the image
    url = from_image_id_to_image_info[image_id]['coco_url']
    file_name = from_image_id_to_image_info[image_id]['file_name']
    save_path = os.path.join(save_root, file_name)
    r = requests.get(url)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    # for novel
    # find the matched bbox for gt bbox "novel"
    all_most_matched_bbox = []
    all_most_matched_bbox_cate = []
    if novel_gt_bboxes.shape[0] != 0:
        #all_predicted_cate = all_proposals[:, -1]
        for novel_bbox in novel_gt_bboxes:
            # load the image
            im = np.array(Image.open(save_path), dtype=np.uint8)
            fig,ax = plt.subplots(1)
            # Display the image
            ax.imshow(im)
            # draw the novel bboxes
            rect = patches.Rectangle((novel_bbox[0], novel_bbox[1]),novel_bbox[2] - novel_bbox[0],novel_bbox[3] - novel_bbox[1],linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
            # find most matched bboxes
            real_iou = iou_calculator(novel_bbox.unsqueeze(dim=0)[:4], all_prediction[:, :4])
            # leave the iou value only when the iou larger than 0
            iou_idx_over_zero = (real_iou > 0)
            # select the top one for each gt bboxes
            if torch.sum(iou_idx_over_zero) == 0:
                continue
            else:
                value, idx = torch.max(real_iou, dim=-1)
                if value < 0.5:
                    continue
                remain_proposal = all_prediction[idx].squeeze(dim=0)
                rect = patches.Rectangle((remain_proposal[0], remain_proposal[1]),remain_proposal[2]-remain_proposal[0],remain_proposal[3]-remain_proposal[1],linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                
                gt_cate = from_cate_id_to_cate_name[novel_bbox[-1].item()]
                pred_cate = from_cate_id_to_cate_name[remain_proposal[-1].item()]

            # save the result
            print_path = os.path.join(save_root, 'printed')
            if not os.path.exists(print_path):
                os.makedirs(print_path)
            temp_file_name = file_name + '_gt_cate_' + gt_cate + '_pred_cate_' + pred_cate + '.jpg'
            plt.savefig(os.path.join(print_path,temp_file_name))
            plt.close()
