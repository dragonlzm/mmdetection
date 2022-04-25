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

#file_path = "C:\\step_0_labeled_datasets(seed=42).json"
#file_path = "C:\\instances_val2017.json"
#file_path = "C:\\step_1_learningloss_newadd.json"
#file_path = "C:\\step_1_influence_newadd.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.proposal.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results_4_by_4.proposal.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\new_rpn_4_by_4_2x.proposal.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.clip_proposal_09.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.clip_proposal_095.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.clip_proposal_09_nms07.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.clip_proposal_09_nms05.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024.patch_acc.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_16_32_512_nms_on_all_07.patch_acc.json"
#proposal_file_path = "/home/zhuoming/results_16_16_1024_nms07.patch_acc.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07.patch_acc.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_16_16_1024_nms07.patch_acc.json"
#proposal_file_path = "/home/zhuoming/results_32_64_1024_nms_on_all_07_nms_over_scales.patch_acc.json"
#proposal_file_path = "/data/zhuoming/detection/test/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_4/results.bbox.json"
proposal_file_path = "/data2/lwll/zhuoming/detection/test/cls_proposal_generator_coco/results_16_16_1024.patch_acc.json"
proposal_result = json.load(open(proposal_file_path))

#gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
#gt_annotation_path = "/data/zhuoming/detection/coco/annotations/instances_val2017.json"
gt_annotation_path = "/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json"
gt_anno_result = json.load(open(gt_annotation_path))


from_img_id_to_pred = {}
from_img_id_to_gt = {}

for ele in proposal_result:
    img_id = ele['image_id']
    bbox = ele['score']
    from_img_id_to_pred[img_id] = bbox
    #bbox = ele['bbox']
    #if img_id not in from_img_id_to_pred:
    #    from_img_id_to_pred[img_id] = []
    #from_img_id_to_pred[img_id].append(bbox)

for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    cate_id = ele['category_id']
    bbox.append(cate_id)
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append(bbox)


#random.shuffle(gt_anno_result['images'])
#first_100_imgs = gt_anno_result['images'][:100]

#all_img_annotation = {info['id']:[] for info in first_100_imgs}
all_img_info = {info['id']:info for info in gt_anno_result['images']}

#for anno_info in gt_anno_result['annotations']:
#    if anno_info['image_id'] in all_img_annotation:
#        all_img_annotation[anno_info['image_id']].append(anno_info)

#save_root = 'C:\\liuzhuoming\\桌面\\images\\random\\'
#save_root = 'C:\\liuzhuoming\\桌面\\images\\influence\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization_4_by_4\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization_4_by_4_2x\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\clip_proposal_05\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\clip_proposal_09\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\clip_proposal_095\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\clip_proposal_09_nms07\\'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_16_32_512_nms_on_all_07\\'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07\\'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_16_16_1024_nms07.patch_acc\\'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_16_16_1024_nms07_novel\\'
#save_root = '/home/zhuoming/results_16_16_1024_nms07_novel/'
#save_root = '/home/zhuoming/results_16_16_1024_nms07_novel_/'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07_novel\\'

save_root = '/home/zhuoming/filtered_bboxes/'
#save_root = '/home/zhuoming/vild_visialization/'

target_list = [397133, 37777, 252219, 87038, 403385, 331352, 386912, 491497, 348881, 289393]
novel_cate_id = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
#target_list = [87871]

iou_calculator = BboxOverlaps2D()

#print(all_img_annotation)
for i, img_id in enumerate(from_img_id_to_pred):
    if img_id not in target_list:
        continue
    #print(type(image_id))
    url = all_img_info[img_id]['coco_url']
    file_name = all_img_info[img_id]['file_name']
    save_path = save_root + file_name
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)  


    im = np.array(Image.open(save_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    
    xyxy_proposal = torch.tensor(from_img_id_to_pred[img_id])

    if img_id in from_img_id_to_gt:
        all_iou_idx = None
        for bbox in from_img_id_to_gt[img_id]:
            cate_id = bbox[-1]
            if cate_id in novel_cate_id:
                rect = patches.Rectangle((bbox[0], bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
                continue
            #bbox = annotation['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
            
            xyxy_gt = torch.tensor([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]])

            # find the proposal 
            real_iou = iou_calculator(xyxy_gt, xyxy_proposal)
            
            # all the bbox that has iou lower than 0.5 will become True
            real_iou_ind = (real_iou < 0.3).view(-1)
            iou_ind = real_iou_ind
            
            if all_iou_idx == None:
                all_iou_idx = iou_ind
            else:
                all_iou_idx = all_iou_idx & iou_ind
        
        remained_bbox = xyxy_proposal[all_iou_idx]
        #remained_bbox = xyxy_proposal
        print(remained_bbox.shape)
            
        for target_proposal in remained_bbox[:100]:
            # draw the proposal
            rect = patches.Rectangle((target_proposal[0], target_proposal[1]),
                                     target_proposal[2]-target_proposal[0],
                                     target_proposal[3]-target_proposal[1], linewidth=1,edgecolor='r',facecolor='none')
            #rect = patches.Rectangle((target_proposal[0], target_proposal[1]),
            #                         target_proposal[2],
            #                         target_proposal[3], linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    #for bbox in from_img_id_to_pred[img_id]:
        #bbox = annotation['bbox']
    #    rect = patches.Rectangle((bbox[0], bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
    #    ax.add_patch(rect)

    #Add the patch to the Axes
    #plt.show()
    #print_path = save_root+'printed\\'
    print_path = save_root+'printed/'
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(print_path+file_name)
    plt.close()

    #if i > 20:
    #    break