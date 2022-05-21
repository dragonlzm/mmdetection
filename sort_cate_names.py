import json

json_content = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017.json'))

from_id_to_name = {ele['id']:ele['name'] for ele in json_content['categories']}
from_name_to_id = {ele['name']:ele['id'] for ele in json_content['categories']}

cate_name_48 = ['person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush']

cate_name_17 = ['airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors']


# cate_name_80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
#            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
#            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
#            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
#            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
#            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
#            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
#            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
#            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
#            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
#            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
#            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
#            'hair drier', 'toothbrush']

# cate_name_60 = ['truck', 'traffic light', 'fire hydrant', 'stop sign',
#                'parking meter', 'bench', 'elephant', 'bear', 'zebra',
#                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
#                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#                'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

# cate_name_20 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
#                 'cow', 'bottle', 'chair', 'couch', 'potted plant',
#                 'dining table', 'tv']

cate_name = cate_name_48 + cate_name_17

cate_id = [from_name_to_id[ele] for ele in cate_name]

cate_id.sort()

print(cate_id)

sorted_name = [from_id_to_name[ele] for ele in cate_id]

print(sorted_name)