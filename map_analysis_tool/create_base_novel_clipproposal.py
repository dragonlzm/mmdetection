# this script aims to create a json file which contains the base,novel and clip proposal for each image
import json
import os
import torch

from matplotlib.font_manager import json_load

# load the gt annotation
#gt_path = '/data/zhuoming/detection/coco/annotations/train_100imgs.json'
gt_path = '/data/zhuoming/detection/coco/annotations/instances_val2017.json'
gt_annotation = json.load(open(gt_path))

base_cates_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_cates_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

from_cate_name_to_cate_id = {ele['name']:ele['id'] for ele in gt_annotation['categories']}
base_cates_id = [from_cate_name_to_cate_id[name] for name in base_cates_name]
novel_cates_id = [from_cate_name_to_cate_id[name] for name in novel_cates_name]

from_image_id_to_bboxes = {}

# split the gt annotation into gt and novel
for anno in gt_annotation['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    bbox = anno['bbox']
    # convert the bbox from xywh to xyxy 
    bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    if image_id not in from_image_id_to_bboxes:
        from_image_id_to_bboxes[image_id] = {'base':[], 'novel':[], 'clip':[], 'base_cates':[], 'novel_cates':[]}
    if cate_id in novel_cates_id:
        from_image_id_to_bboxes[image_id]['novel'].append(bbox)
        from_image_id_to_bboxes[image_id]['novel_cates'].append(cate_id)
    elif cate_id in base_cates_id:
        from_image_id_to_bboxes[image_id]['base'].append(bbox)
        from_image_id_to_bboxes[image_id]['base_cates'].append(cate_id)

# load the clip proposal
clip_proposal_path = '/data/zhuoming/detection/coco/clip_proposal/32_32_512'
save_path = '/home/zhuoming/base_novel_clippro'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for img_info in gt_annotation['images']:
    image_name = img_info['file_name']
    image_id = img_info['id']
    json_file_name = image_name.split('.')[0] + '.json'
    random_file_path = os.path.join(clip_proposal_path, json_file_name) 
    # load the clip proposal
    if not os.path.exists(random_file_path):
        pregenerated_bbox = []
    else:
        clip_proposal_file = json.load(open(random_file_path))
        pregenerated_bbox = clip_proposal_file['score']
    if image_id not in from_image_id_to_bboxes:
        from_image_id_to_bboxes[image_id] = {'base':[], 'novel':[], 'clip':[], 'base_cates':[], 'novel_cates':[]}
    from_image_id_to_bboxes[image_id]['clip'] = pregenerated_bbox
    save_file = os.path.join(save_path, json_file_name)
    file = open(save_file, 'w')
    file.write(json.dumps(from_image_id_to_bboxes[image_id]))
    file.close()