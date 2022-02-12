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

#save_path = 'C:\\Users\\XPS\\Desktop\\174228.jpg'
#save_path = 'C:\\Users\\XPS\\Desktop\\472054.jpg'
#save_path = 'C:\\Users\\XPS\\Desktop\\397133.jpg'
save_path = 'C:\\Users\\XPS\\Desktop\\174482.jpg'


#save_path = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\large_person\\000000554002.jpg'
#save_path = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000507037.jpg'
#save_path = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000580410.jpg'
#save_path = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000275198.jpg'
im = np.array(Image.open(save_path), dtype=np.uint8)
height, width, c = im.shape

'''
# for random bboxes
proposal_file_path = "C:\\Users\\XPS\\Desktop\\random_bbox_ratio.npy"
proposal_result = np.load(proposal_file_path)

all_bbox = []
for ratio in proposal_result:
    x = ratio[0] * width
    y = ratio[1] * height
    w = ratio[2] * width - x
    h = ratio[3] * height - y
    bbox = [x, y, w, h]
    all_bbox.append(bbox)'''

gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_val2017.json"
#gt_annotation_path = "C:\\Users\\XPS\\Desktop\\annotations\\instances_train2017.json"
gt_anno_result = json.load(open(gt_annotation_path))

from_img_id_to_gt = {}

# collecting the gt bboxes
for ele in gt_anno_result['annotations']:
    img_id = ele['image_id']
    bbox = ele['bbox']
    if img_id not in from_img_id_to_gt:
        from_img_id_to_gt[img_id] = []
    from_img_id_to_gt[img_id].append(bbox)

#if img_id in from_img_id_to_gt:
#    for box in from_img_id_to_gt[img_id]:
#        #box = annotation['bbox']
#        rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='g',facecolor='none')
#        ax.add_patch(rect)

#save_root = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000507037\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000580410\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\random_bbox\\000000275198\\'

#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000275198\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000580410\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000507037\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000554002\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000174228\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000472054\\'
#save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000397133\\'
save_root = 'C:\\Users\\XPS\\Desktop\\gt_bbox\\000000174482\\'

#for i, box in enumerate(all_bbox):
#for i, box in enumerate(from_img_id_to_gt[275198]):
#for i, box in enumerate(from_img_id_to_gt[580410]):  
#for i, box in enumerate(from_img_id_to_gt[507037]):
#for i, box in enumerate(from_img_id_to_gt[554002]): 
#from_img_id_to_gt[174228].append([278.87, 126.33, 150, 216.99])
#from_img_id_to_gt[174228].append([278.87, 126.33, 170, 216.99])
#from_img_id_to_gt[174228].append([278.87, 126.33, 190, 216.99])
#from_img_id_to_gt[174228].append([278.87, 126.33, 210, 216.99])
#from_img_id_to_gt[472054].append([302.25, 90.13, 130.6, 326.66])
#from_img_id_to_gt[472054].append([282.25, 90.13, 150.6, 326.66])
#from_img_id_to_gt[472054].append([262.25, 90.13, 170.6, 326.66])
#from_img_id_to_gt[472054].append([242.25, 90.13, 190.6, 326.66])
#from_img_id_to_gt[472054].append([222.25, 90.13, 210.6, 326.66])

#[357.29, 126.33, 282.71, 216.99]
#[278.87, 155.85, 136.1, 199.68]
#for i, box in enumerate(from_img_id_to_gt[174228]):
#for i, box in enumerate(from_img_id_to_gt[472054]):  
#for i, box in enumerate(from_img_id_to_gt[397133]):     
for i, box in enumerate(from_img_id_to_gt[174482]):
    #box = annotation['bbox']
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    rect = patches.Rectangle((box[0], box[1]),box[2],box[3],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    print(box)

    #Add the patch to the Axes
    #plt.show()
    print_path = save_root
    if not os.path.exists(print_path):
        os.makedirs(print_path)
    #file_name = '000000507037_' + str(i) + '.jpg'
    #file_name = '000000580410_' + str(i) + '.jpg'
    #file_name = '000000275198_' + str(i) + '.jpg' 
    #file_name = '000000554002_' + str(i) + '.jpg' 
    #file_name = '000000174228_' + str(i) + '.jpg'    
    #file_name = '000000472054_' + str(i) + '.jpg'
    #file_name = '000000397133_' + str(i) + '.jpg'
    file_name = '000000174482_' + str(i) + '.jpg'
         
    plt.savefig(print_path+file_name)
    plt.close()