# this script aims to visualize the bbox of FP in the "filtered_fp_in_prediction"
# in filtered_fp_in_prediction the fp bboxes will be the one which with confidence score 0

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

proposal_file_path = "/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/filter_fp_prediction.json"
proposal_result = json.load(open(proposal_file_path))

gt_annotation_path = "/data/zhuoming/detection/coco/annotations/instances_val2017.json"
gt_anno_result = json.load(open(gt_annotation_path))

from_img_id_to_pred = {}
from_img_id_to_gt = {}

# the proposal should like this
#{'image_id': 397133, 'bbox': [381.830322265625, 68.92253875732422, 120.26748657226562, 282.2848815917969], 'score': 1.0, 'category_id': 1}
for ele in proposal_result:
    img_id = ele['image_id']
    bbox = ele['bbox']
    score = ele['score']
    category_id = ele['category_id']
    if score == 1:
        continue
    if img_id not in from_img_id_to_pred:
        from_img_id_to_pred[img_id] = []
    from_img_id_to_pred[img_id].append((bbox,category_id))

for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    category_id = ele['category_id']
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append((bbox, category_id))

#random.shuffle(gt_anno_result['images'])
#first_100_imgs = gt_anno_result['images'][:100]

#all_img_annotation = {info['id']:[] for info in first_100_imgs}
all_img_info = {info['id']:info for info in gt_anno_result['images']}

from_cate_id_to_cate_name = {ele['id']: ele['name'] for ele in gt_anno_result['categories']}

#save_root = '/home/zhuoming/test_min_conf/'
save_root = '/home/zhuoming/visualize_fp/'

#target_list = [526043, 124756, 186317, 256607, 546742, 87871, 397543, 555574, 30068, 89549, 260910, 299325]

#print(all_img_annotation)
for i, img_id in enumerate(from_img_id_to_pred):
    if i > 20:
        break
    #if img_id not in target_list:
    #    continue
    #print(type(image_id))
    url = all_img_info[img_id]['coco_url']
    file_name = all_img_info[img_id]['file_name']
    save_path = save_root + file_name
    r = requests.get(url)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    with open(save_path, 'wb') as f:
        f.write(r.content)  

    for cate in from_cate_id_to_cate_name:
        im = np.array(Image.open(save_path), dtype=np.uint8)
        fig,ax = plt.subplots(1)

        # Display the image
        ax.imshow(im)

        if img_id in from_img_id_to_gt:
            for box, cate_id in from_img_id_to_gt[img_id]:
                if cate_id != cate:
                    continue
                #box = annotation['bbox']
                rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
                ax.add_patch(rect)

        for box, cate_id in from_img_id_to_pred[img_id]:
            if cate_id != cate:
                continue
            #box = annotation['bbox']
            rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

        #Add the patch to the Axes
        #plt.show()
        print_path = os.path.join(save_root, 'printed')
        if not os.path.exists(print_path):
            os.makedirs(print_path)

        splited_name = file_name.split('.')
        now_file_name = '.'.join(splited_name[:-1]) + "_" + from_cate_id_to_cate_name[cate] + '.' + splited_name[-1]
        plt.savefig(os.path.join(print_path, now_file_name))
        plt.close()