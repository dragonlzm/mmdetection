# this script aims to generate the dataset for proposal selection experiment
import json

train_annotations_file = '/data/zhuoming/detection/coco/annotations/instances_train2017.json'
# for the training set of this experiment would be the image that only has the base categories
train_annotations_content = json.load(open(train_annotations_file))

# aggregate the annoations base on the image id
# aggregate the cate_id base on the image id
from_image_id_to_annotaions = {}
from_image_id_to_cates = {}

for anno in train_annotations_content['annotations']:
    image_id = anno['image_id']
    category_id = anno['category_id']
    bbox = anno['bbox']
    if image_id not in from_image_id_to_annotaions:
        from_image_id_to_annotaions[image_id] = []
    from_image_id_to_annotaions[image_id].append(bbox)
    if image_id not in from_image_id_to_cates:
        from_image_id_to_cates[image_id] = []
    from_image_id_to_cates[image_id].append(category_id)

# make the cate_id unique
from_image_id_to_unique_cates = {key:set(from_image_id_to_cates[key]) for key in from_image_id_to_cates}

# create the set for categories id which is not belong to the image 
cate_name_48 = ['person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush']

from_cate_name_to_cate_id = {anno['name']:anno['id'] for anno in train_annotations_content['categories']}
all_cate_name = [anno['name'] for anno in train_annotations_content['categories']]

except_base_id = set([from_cate_name_to_cate_id[name] for name in all_cate_name if name not in cate_name_48])
base_id = set([from_cate_name_to_cate_id[name] for name in cate_name_48])

# prepare the image info
from_image_id_to_image_info = {info['id']:info for info in train_annotations_content['images']}

all_base_only_annos = []
all_base_only_image_info = []

all_other_annos = []
all_other_image_info = []
# use the set intersect between the two set and check whether the image has the non-base categories 
# for the validation set is the rest of the image
for image_id in from_image_id_to_annotaions:
    all_cates_on_image = from_image_id_to_unique_cates[image_id]
    except_cates_intersect = except_base_id.intersection(all_cates_on_image)
    base_cates_intersect = base_id.intersection(all_cates_on_image)
    # if has the base cates but does not has any other cates
    if len(except_cates_intersect) == 0 and len(base_cates_intersect) != 0:
        all_base_only_annos += from_image_id_to_annotaions[image_id]
        all_base_only_image_info.append(from_image_id_to_image_info[image_id])
    else:
        all_other_annos += from_image_id_to_annotaions[image_id]
        all_other_image_info.append(from_image_id_to_image_info[image_id])

base_only_json_content = {'info':train_annotations_content['info'], 'licenses':train_annotations_content['licenses'], 
                          'images':all_base_only_image_info, 'annotations':all_base_only_annos, 'categories':train_annotations_content['categories']}
except_base_only_json_content = {'info':train_annotations_content['info'], 'licenses':train_annotations_content['licenses'], 
                          'images':all_other_image_info, 'annotations':all_other_annos, 'categories':train_annotations_content['categories']}

print('base48 only images num:', len(all_base_only_image_info), 'base48 only anno num:', len(all_base_only_annos))
print('except base48 only images num:', len(all_other_image_info), 'except base48 only anno num:', len(all_other_annos))


file = open('/data/zhuoming/detection/coco/annotations/instances_train2017_48base_only.json', 'w')
file.write(json.dumps(base_only_json_content))
file.close()

file = open('/data/zhuoming/detection/coco/annotations/instances_train2017_except_48base_only.json', 'w')
file.write(json.dumps(except_base_only_json_content))
file.close()