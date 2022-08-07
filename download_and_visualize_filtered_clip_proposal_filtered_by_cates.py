# this script aims to visualize the proposal before filtering, which is generated by the clip_proposal_generator
# and also visualize the proposal after filtering the base categories. The file also has the feature in it
# give out two root path of the model
import os
import json
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from os import listdir
from os.path import isfile, join


#file_list = ['000000000113.json']

original_proposal_file_root = '/project/nevatia_174/zhuoming/detection/coco/clip_proposal/32_32_512'
proposal_with_feat_file_root = '/project/nevatia_174/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/random'


file_list = [f for f in listdir(proposal_with_feat_file_root) if isfile(join(proposal_with_feat_file_root, f))]

# need the gt annotations:
gt_annotation_path = '/project/nevatia_174/zhuoming/detection/coco/annotations/instances_train2017.json'
gt_content = json.load(open(gt_annotation_path))
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

save_root = '/home1/liuzhuom/filtered_visualization'

# convert the file_list from image name to image id
file_id_list = [int(ele.split('.')[0].lstrip('0')) for ele in file_list]

for i, (file, file_id) in enumerate(zip(file_list, file_id_list)):
    print(file, file_id)
    if i > 100:
        break
    # load the original proposal
    original_proposal_file_name = os.path.join(original_proposal_file_root, file)
    original_proposal_file_content = json.load(open(original_proposal_file_name))
    # dict_keys(['image_id', 'score']), original_proposal_file_content['score'] 1000, 5
    original_proposals = torch.tensor(original_proposal_file_content['score'])
    
    # load the filter proposal with feat
    proposal_with_feat_file_name = os.path.join(proposal_with_feat_file_root, file)
    proposal_with_feat_file_content = json.load(open(proposal_with_feat_file_name))
    filtered_proposals = np.array(proposal_with_feat_file_content['bbox'])
    filtered_proposals = filtered_proposals[:, :4]
    # scale the bbox back to original size
    pre_extract_scale_factor = np.array(proposal_with_feat_file_content['img_metas']['scale_factor']).astype(np.float32)
    filtered_proposals = filtered_proposals / pre_extract_scale_factor
    
    # download the image 
    url = from_image_id_to_image_info[file_id]['coco_url']
    file_name = from_image_id_to_image_info[file_id]['file_name']
    save_path = os.path.join(save_root, file_name)
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)

    # visualize the before filter
    im = np.array(Image.open(save_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    # print the base
    for box in from_image_id_to_annotation[file_id]['base']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    # print the novel
    for box in from_image_id_to_annotation[file_id]['novel']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # print the prediction
    all_prediction = original_proposals
    # select top 50    
    all_prediction = all_prediction[:20]

    for box in all_prediction:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed_before_filter')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()
    
    
    # viuslize after filter
    im = np.array(Image.open(save_path), dtype=np.uint8)
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    # print the base
    for box in from_image_id_to_annotation[file_id]['base']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    # print the novel
    for box in from_image_id_to_annotation[file_id]['novel']:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # print the prediction
    all_prediction = filtered_proposals
    # select top 50    
    all_prediction = all_prediction[:20]

    #print('testing after filter', len(all_prediction))
    for box in all_prediction:
        #print(box)
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed_after_filter')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()