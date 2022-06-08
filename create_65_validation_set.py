# this script aims to create the validation set which only has base and novel categories

novel_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

base_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

import json
original_valiation_set = json.load(open('instances_val2017.json'))

# obtain the categories id
from_name_to_id = {}
for anno in original_valiation_set['categories']:
    name = anno['name']
    cate_id = anno['id']
    from_name_to_id[name] = cate_id
all_65_cates_id = [from_name_to_id[name] for name in (novel_name + base_name)]    

# obtain the annotation
from_image_id_to_anno = {}
for anno in original_valiation_set['annotations']:
    cate_id = anno['category_id']
    image_id = anno['image_id']
    if cate_id not in all_65_cates_id:
        continue
    if image_id not in from_image_id_to_anno:
        from_image_id_to_anno[image_id] = []
    from_image_id_to_anno[image_id].append(anno)

# obtain the image info
from_image_id_to_image_anno = {anno['id']:anno for anno in original_valiation_set['images']}

all_annotations = []
for image_id in from_image_id_to_anno:
    all_annotations += from_image_id_to_anno[image_id]

all_image_info = []
for image_id in from_image_id_to_anno:
    all_image_info.append(from_image_id_to_image_anno[image_id])

final_res = {'info':original_valiation_set['info'], 
             'licenses':original_valiation_set['licenses'], 
             'images':all_image_info, 
             'annotations':all_annotations, 
             'categories':original_valiation_set['categories']}

file = open('instances_val2017_65cates.json', 'w')
file.write(json.dumps(final_res))
file.close()
