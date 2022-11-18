# this script aims to analyse the dataset statistic
import json

# load the annotation
#dataset_path = '/data/zhuoming/detection/coco/annotations/instances_train2017.json'
dataset_path = '/data/zhuoming/detection/coco/annotations/instances_val2017.json'
#dataset_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_train.json'
#dataset_path = '/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_val.json'
dataset_content = json.load(open(dataset_path))

# present the number of total image 
print('number of image:', len(dataset_content['images']))

# define the base and novel categories name
# base_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
# novel_cate_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard', 'sink', 'scissors')

base_cate_name = ('truck', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
            'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush')
novel_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 
                    'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
                    'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
                    'couch', 'potted plant', 'dining table', 'tv')


# obtain the conversion from the name to the cate_id
from_cate_name_to_cate_id = {ele['name']:ele['id'] for ele in dataset_content['categories']}
# obtain the base_id and the novel_id
base_cate_id = [from_cate_name_to_cate_id[name] for name in base_cate_name]
novel_cate_id = [from_cate_name_to_cate_id[name] for name in novel_cate_name]

# aggregate the annotation base on the image
from_img_id_to_annotation = {}
for anno in dataset_content['annotations']:
    image_id = anno['image_id']
    if image_id not in from_img_id_to_annotation:
        from_img_id_to_annotation[image_id] = {'base':[], 'novel':[]}
    category_id = anno['category_id']
    bbox = anno['bbox']
    #bbox_size = bbox[2] * bbox[3]
    if category_id in base_cate_id:
        from_img_id_to_annotation[image_id]['base'].append(bbox)
    elif category_id in novel_cate_id:
        from_img_id_to_annotation[image_id]['novel'].append(bbox)

# for train set
# base_image_id = [key for key in from_img_id_to_annotation if len(from_img_id_to_annotation[key]['base']) != 0]
# print('base_image_num', len(base_image_id))

# base_cate_anno = []
# for key in from_img_id_to_annotation:
#     if len(from_img_id_to_annotation[key]['base']) != 0:
#         base_cate_anno += from_img_id_to_annotation[key]['base']

# print('base_cate_instance', len(base_cate_anno))


# for test set
image_num = [key for key in from_img_id_to_annotation if len(from_img_id_to_annotation[key]['base']) != 0 or len(from_img_id_to_annotation[key]['novel']) != 0]
print('image_num', len(image_num))

base_cate_anno = []
for key in from_img_id_to_annotation:
    if len(from_img_id_to_annotation[key]['base']) != 0:
        base_cate_anno += from_img_id_to_annotation[key]['base']

print('base_cate_instance', len(base_cate_anno))


novel_cate_anno = []
for key in from_img_id_to_annotation:
    if len(from_img_id_to_annotation[key]['novel']) != 0:
        base_cate_anno += from_img_id_to_annotation[key]['novel']

print('novel_cate_instance', len(base_cate_anno))