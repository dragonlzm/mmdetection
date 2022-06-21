# this script aims to visualize the sorted proposal 
# and visualize all the proposal which has the highest iou with each gt bbox
import json
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# load the gt bboxes
gt_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json'))

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
base_cates_ids = [from_cate_name_to_cate_id[name] for name in base_cates_name]

# aggreate the anotation base on the image id
from_image_id_to_annotation = {}
for anno in gt_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
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

# load the sortting info
#sortting_score = json.load(open('/data/zhuoming/detection/test/proposal_selection_v1/testing.gt_acc.json'))
#sortting_score = json.load(open('/data/zhuoming/detection/test/proposal_selector_coco_with_feat/testing.gt_acc.json'))
#sortting_score = json.load(open('/data/zhuoming/detection/test/proposal_selector_coco_v3/testing.gt_acc.json'))
sortting_score = json.load(open('/data/zhuoming/detection/test/proposal_selector_coco_with_feat_v2/testing.gt_acc.json'))

from_image_id_to_sortting_idx = {}
for ele in sortting_score:
    image_id = ele['image_id']
    scores = torch.tensor(ele['score'][0])
    sorted_val, sorted_idx = torch.sort(scores, descending=True)
    from_image_id_to_sortting_idx[image_id] = sorted_idx    

# load the proposal and print the image
#save_root = '/home/zhuoming/sorted_proposal'
#save_root = '/home/zhuoming/sorted_proposal_v2'
save_root = '/home/zhuoming/sorted_proposal_with_feat_v2'
proposal_path_root = 'data/coco/clip_proposal/32_32_512'

for i, image_id in enumerate(from_image_id_to_annotation):
    if i > 50:
        break
    #print(type(image_id))
    url = from_image_id_to_image_info[image_id]['coco_url']
    file_name = from_image_id_to_image_info[image_id]['file_name']
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
    # print the proposal 
    pregenerate_prop_path = os.path.join(proposal_path_root, '.'.join(file_name.split('.')[:-1]) + '.json')
    pregenerated_bbox = json.load(open(pregenerate_prop_path))
    all_proposals = torch.tensor(pregenerated_bbox['score'])[:,:4]
    sorted_proposal = all_proposals[from_image_id_to_sortting_idx[image_id]]
    
    # select top 50
    sorted_proposal = sorted_proposal[:20]
    for box in sorted_proposal:
        rect = patches.Rectangle((box[0], box[1]),box[2]-box[0],box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()













# if the iou of one gt bboxes with all proposals are equal to 0, then we should not draw any proposal
# for this bboxes