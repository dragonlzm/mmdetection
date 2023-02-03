# this script aim to visualize the bbox, using the clip feat file.
import json
import random
import requests
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch

clip_feat_path = '/data/zhuoming/detection/coco/clip_bg_proposal_feat/base48_finetuned/random'
onlyfiles = [os.path.join(clip_feat_path, f) for f in os.listdir(clip_feat_path) if os.path.isfile(os.path.join(clip_feat_path, f))]
selected_file = onlyfiles[:100]

# load the gt first
gt_annotation_path = "/data/zhuoming/detection/coco/annotations/instances_train2017.json"
gt_anno_result = json.load(open(gt_annotation_path))
from_img_id_to_gt = {}
from_image_name_to_image_id = {}

for ele in gt_anno_result['annotations']:
    image_id = ele['image_id']
    bbox = ele['bbox']
    if image_id not in from_img_id_to_gt:
        from_img_id_to_gt[image_id] = []
    from_img_id_to_gt[image_id].append(bbox)

for ele in gt_anno_result['images']:
    image_id = ele['id']
    image_name = ele['file_name']
    from_image_name_to_image_id[image_name] = image_id

# load the predict
from_img_id_to_pred = {}
for file in selected_file:
    proposal_result = json.load(open(file))
    
    bbox = torch.tensor(proposal_result['bbox'])
    scale_factor = torch.tensor(proposal_result['img_metas']['scale_factor'])
    image_name = proposal_result['img_metas']['ori_filename']
    image_id = from_image_name_to_image_id[image_name]
    
    bbox = bbox[:, :4]
    bbox /= bbox.new_tensor(scale_factor)
    from_img_id_to_pred[image_id] = bbox
    
all_img_info = {}
for ele in gt_anno_result['images']:
    image_id = ele['id']
    if image_id in from_img_id_to_pred:
        all_img_info[image_id] = ele
        
save_root = '/home/zhuoming/dc_clipproposal'


#print(all_img_annotation)
for img_id in all_img_info:
    #print(type(image_id))
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
    # draw the gt
    if img_id in from_img_id_to_gt:
        for box in from_img_id_to_gt[img_id]:
            #box = annotation['bbox']
            rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
            ax.add_patch(rect)
    # draw the proposal the format should be xyxy 
    for box in from_img_id_to_pred[img_id]:
        #box = annotation['bbox']
        rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    #Add the patch to the Axes
    #plt.show()
    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()