# this script aims to generate a subset of dataset in which each image contains novel categories'
# instances, the annotation for each image also contain the annotation from base categories
from itertools import count
import json

annotation_file = "/data/zhuoming/detection/coco/annotations/instances_train2017.json"
annotation_content = json.load(open(annotation_file))
novel_cate_id = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]

# aggreate image id base on category ids
from_cate_id_to_image_id = {}
for ele in annotation_content['annotations']:
    category_id = ele['category_id']
    image_id = ele['image_id']
    if category_id not in from_cate_id_to_image_id:
        from_cate_id_to_image_id[category_id] = []
    from_cate_id_to_image_id[category_id].append(image_id)

# deduplicate the image id for each cate
for category_id in from_cate_id_to_image_id:
    from_cate_id_to_image_id[category_id] = list(set(from_cate_id_to_image_id[category_id]))

# select the category id of novel categories, evenly
selected_img_list = []
for novel_id in novel_cate_id:
    novel_img_list = from_cate_id_to_image_id[category_id]
    count = 0
    for image_id in novel_img_list:
        if image_id not in selected_img_list:
            selected_img_list.append(image_id)
            count += 1
        else:
            continue
        if count >= 10:
            break
    print(novel_id, count)

print(len(selected_img_list))

# obtain the image information and the annotation information for each image
needed_image_info = []
needed_anno_info = []

for ele in annotation_content['images']:
    image_id = ele['id']
    if image_id in selected_img_list:
        needed_image_info.append(ele)

for ele in annotation_content['annotations']:
    image_id = ele['image_id']
    if image_id in selected_img_list:
        needed_anno_info.append(ele)

all_result = {'info':annotation_content['info'], 'annotations':needed_anno_info, 'images':needed_image_info, 'licenses':annotation_content['licenses'], 'categories':annotation_content['categories']}

file = open('coco_10img_per_novel.json', 'w')
file.write(json.dumps(all_result))
file.close()