# this script aims to visualize the top K prediction(sorted by confidence score)
import json
import os
import torch
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# load the prediction file
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_and_novel.bbox.json'
#file_path = '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48/base_results.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_and_novel.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_and_novel_e18.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json'
#file_path = '/data/zhuoming/detection/mask_rcnn_clip_classifier/results_base48.bbox.json'
file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight/base_and_novel.bbox.json'

pred_content = json.load(open(file_path))
# aggregate the predition base on the image
from_image_id_to_prediction = {}
for res in pred_content:
    image_id = res['image_id']
    bbox = res['bbox']
    score = res['score']
    if image_id not in from_image_id_to_prediction:
        from_image_id_to_prediction[image_id] = {'bboxes':[], 'scores':[]}
    from_image_id_to_prediction[image_id]['bboxes'].append(bbox)
    from_image_id_to_prediction[image_id]['scores'].append(score)

# sort the prediction base on the score
for image_id in from_image_id_to_prediction:
    all_bboxes_per_img = torch.tensor(from_image_id_to_prediction[image_id]['bboxes'])
    all_scores_per_img = torch.tensor(from_image_id_to_prediction[image_id]['scores'])
    sorted_scores, sorted_indices = torch.sort(all_scores_per_img, descending=True)
    sorted_bboxes = all_bboxes_per_img[sorted_indices]
    from_image_id_to_prediction[image_id] = {'bboxes':sorted_bboxes, 'scores':sorted_scores}


# load the gt bboxes
gt_anno_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'

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
    
#save_root = '/home/zhuoming/mcnn_baseline_2x_top50_prediction'
#save_root = '/home/zhuoming/mcnn_2x_e24_top50_prediction'
#save_root = '/home/zhuoming/mcnn_2x_e18_top50_prediction'
#save_root = '/home/zhuoming/mcnn_1x_e12_top50_prediction_base48'
#save_root = '/home/zhuoming/mcnn_1x_e12_top20_prediction_base48'
save_root = '/home/zhuoming/download_and_visualize_topk_prediction'

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
    # print the prediction
    all_prediction = from_image_id_to_prediction[image_id]['bboxes']
    # select top 50    
    all_prediction = all_prediction[:20]

    for box in all_prediction:
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)    

    print_path = os.path.join(save_root, 'printed')
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(os.path.join(print_path,file_name))
    plt.close()