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

gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
gt_anno_result = json.load(open(gt_annotation_path))

from_img_id_to_gt = {}

all_img_info = {info['id']:info for info in gt_anno_result['images']}

# collecting the gt bboxes and their categories
for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    cate_id = ele['category_id']
    bbox.append(cate_id)
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append(bbox)

#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\median_person\\'

#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\fork\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\spoon\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\oven\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\remote\\'
save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\bowl\\'

visual_num = 0

for img_id in from_img_id_to_gt.keys():
    # search whether there is a satisfying bbox
    has_target = False
    target_bboxes = []
    for i, bbox in enumerate(from_img_id_to_gt[img_id]):
        cate = bbox[-1]
        area = bbox[2] * bbox[3]
        #if bbox[-1] == 1 and area > 96 ** 2:
        #if bbox[-1] == 1 and area < 96 ** 2 and area > 32 ** 2:
        #if bbox[-1] == 48:
        #if bbox[-1] == 50:
        #if bbox[-1] == 79:
        #if bbox[-1] == 75:  
        if bbox[-1] == 51:       
            has_target = True
            target_bboxes.append(bbox)
    if has_target:
        visual_num += 1
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
        for bbox in target_bboxes:
            rect = patches.Rectangle((bbox[0], bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)

        #Add the patch to the Axes
        #plt.show()
        print_path = save_root + 'print\\'
        if not os.path.exists(print_path):
            os.makedirs(print_path)
        #file_name = '000000507037_' + str(i) + '.jpg'
        #file_name = '000000580410_' + str(i) + '.jpg'
        file_name = str(img_id) + '_' + str(i) + '.jpg'    
        plt.savefig(print_path+file_name)
        plt.close()
    if visual_num > 20:
        break