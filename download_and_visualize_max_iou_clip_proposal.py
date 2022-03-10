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
proposal_file_path = "/home/zhuoming/results_16_16_1024_nms07.patch_acc.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07.patch_acc.json"
#proposal_file_path = "C:\\Users\\Zhuoming Liu\\Desktop\\results_16_16_1024_nms07.patch_acc.json"


proposal_result = json.load(open(proposal_file_path))

#gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
gt_annotation_path = "/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json"
gt_anno_result = json.load(open(gt_annotation_path))


from_img_id_to_pred = {}
from_img_id_to_gt = {}

for ele in proposal_result:
    img_id = ele['image_id']
    bbox = ele['score']
    #if img_id not in from_img_id_to_pred:
    from_img_id_to_pred[img_id] = bbox
    #from_img_id_to_pred[img_id].append(bbox)

for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
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
save_root = '/home/zhuoming/results_16_16_1024_nms07_novel/'
#save_root = '/home/zhuoming/results_16_16_1024_nms07_novel_/'
#save_root = 'C:\\Users\\Zhuoming Liu\\Desktop\\results_32_64_1024_nms07_novel\\'

target_list = [526043, 124756, 186317, 256607, 546742, 87871, 397543, 555574, 30068, 89549, 260910, 299325]
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
        for box in from_img_id_to_gt[img_id]:
            #box = annotation['bbox']
            rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
            
            xyxy_gt = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]]])
            #print('xyxy_gt', xyxy_gt)
            
            # find the proposal 
            iou = iou_calculator(xyxy_gt, xyxy_proposal)
            #print('iou', iou)
            max_id = torch.max(iou, dim=-1)[1].item()
            #print('max_id', max_id)
            #target_proposal = from_img_id_to_pred[img_id][max_id]
            target_proposal = xyxy_proposal[max_id]
            #print('target_proposal', target_proposal)
            #print('from_img_id_to_pred[img_id][max_id]', from_img_id_to_pred[img_id][max_id])
            
            # draw the proposal
            rect = patches.Rectangle((target_proposal[0], target_proposal[1]),
                                     target_proposal[2]-target_proposal[0],
                                     target_proposal[3]-target_proposal[1], linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

    #for box in from_img_id_to_pred[img_id]:
        #box = annotation['bbox']
    #    rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
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