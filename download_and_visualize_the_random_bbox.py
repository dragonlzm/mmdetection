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
import torch

#file_path = "C:\\step_0_labeled_datasets(seed=42).json"
#file_path = "C:\\instances_val2017.json"
#file_path = "C:\\step_1_learningloss_newadd.json"
#file_path = "C:\\step_1_influence_newadd.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results.proposal.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\results_4_by_4.proposal.json"
#proposal_file_path = "C:\\Users\\XPS\\Desktop\\new_rpn_4_by_4_2x.proposal.json"
proposal_file_path = "C:\\Users\\XPS\\Desktop\\random_bbox_ratio.npy"
proposal_result = np.load(proposal_file_path)

gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
gt_anno_result = json.load(open(gt_annotation_path))


from_img_id_to_pred = {}
from_img_id_to_gt = {}

# collecting the gt bboxes
for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append(bbox)

from_img_id_to_rand = {}
# generating the random_bboxes
for ele in gt_anno_result['images']:
    img_id = ele['id']
    width, height = ele['width'], ele['height']
    from_img_id_to_rand[img_id] = []
    for ratio in proposal_result:
        x = ratio[0] * width
        y = ratio[1] * height
        w = ratio[2] * width - x
        h = ratio[3] * height - y
        bbox = [x, y, w, h]
        from_img_id_to_rand[img_id].append(bbox)
    

random.shuffle(gt_anno_result['images'])
first_100_imgs = gt_anno_result['images'][:100]

#all_img_annotation = {info['id']:[] for info in first_100_imgs}
all_img_info = {info['id']:info for info in first_100_imgs}



#for anno_info in gt_anno_result['annotations']:
#    if anno_info['image_id'] in all_img_annotation:
#        all_img_annotation[anno_info['image_id']].append(anno_info)


#save_root = 'C:\\liuzhuoming\\桌面\\images\\random\\'
#save_root = 'C:\\liuzhuoming\\桌面\\images\\influence\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization_4_by_4\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\visualization_4_by_4_2x\\'
save_root = 'C:\\Users\\XPS\\Desktop\\random_bbox\\'

#print(all_img_annotation)
for img_id in all_img_info:
    print(img_id)
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

    #if img_id in from_img_id_to_gt:
    #    for box in from_img_id_to_gt[img_id]:
    #        #box = annotation['bbox']
    #        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
    #        ax.add_patch(rect)

    for box in from_img_id_to_rand[img_id]:
        #box = annotation['bbox']
        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    #Add the patch to the Axes
    #plt.show()
    print_path = save_root+'printed\\'
    if not os.path.exists(print_path):
        os.makedirs(print_path)

    plt.savefig(print_path+file_name)
    plt.close()