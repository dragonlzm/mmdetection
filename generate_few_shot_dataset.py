import json
import random

# fix the random seed
random.seed(42)

#bbox_per_cate = 100
bbox_per_cate = 10

json_content = json.load(open('/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json'))
selected_image = {}
categories_count = {}

# go through all the categories to take the notes
for cate_info in json_content['categories']:
    cate_id = cate_info['id']
    categories_count[cate_id] = 0

selected_annotation = []
# for each image we only select one bbox
# go through all the annotation in the json file
for anno in json_content['annotations']:
    image_id = anno['image_id']
    cate_id = anno['category_id']
    # if the image has been selected, skip the annotation
    if image_id in selected_image:
        continue
    # if the cate is full, skip the annotation
    if categories_count[cate_id] >= bbox_per_cate:
        continue 
    # add the annotation to the list
    selected_annotation.append(anno)
    categories_count[cate_id] += 1
    selected_image[image_id] = 1

# shuffle the annotations
random.shuffle(selected_annotation)

# filter the image annotations
selected_image_anno = []
for image_anno in json_content['images']:
    image_id = image_anno['id']
    if image_id in selected_image:
        selected_image_anno.append(image_anno)

print('image_anno', len(selected_image_anno))
print('selected_annotation', len(selected_annotation))

result_json_content = {key:json_content[key] for key in json_content}
result_json_content['images'] = selected_image_anno
result_json_content['annotations'] = selected_annotation

#file = open('/data2/lwll/zhuoming/detection/coco/annotations/train_100shots.json', 'w')
file = open('/data2/lwll/zhuoming/detection/coco/annotations/train_10shots.json', 'w')
file.write(json.dumps(result_json_content))
file.close()