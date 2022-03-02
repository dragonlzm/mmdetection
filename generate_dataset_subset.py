import json
import random
from select import select

# fix the random seed
random.seed(42)

json_content = json.load(open('/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json'))

num_img = 100
img_list = json_content['images']
random.shuffle(img_list)

selected_img = img_list[:100]
img_id_dict = {ele['id']:1 for ele in selected_img}

selected_annotation = []
for anno in json_content['annotations']:
    img_id = anno['image_id']
    if img_id in img_id_dict:
        selected_annotation.append(anno)

['info', 'licenses', 'images', 'annotations', 'categories']

result_json = {'info': json_content['info'], 'licenses':json_content['licenses'], 
 'categories':json_content['categories']}

result_json['images'] = selected_img
result_json['annotations'] = selected_annotation

file = open('/data2/lwll/zhuoming/detection/coco/annotations/train_100imgs.json', 'w')
file.write(json.dumps(result_json))
file.close()