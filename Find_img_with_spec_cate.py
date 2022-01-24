import json

path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'

json_cont = json.load(open(path))

from_cate_id_to_img_id = {}

for ele in json_cont['categories']:
    this_id = ele['id']
    from_cate_id_to_img_id[this_id] = []

for ele in json_cont['annotations']:
    img_id = ele['image_id']
    cate_id = ele['category_id']
    if img_id not in from_cate_id_to_img_id[cate_id]:
        from_cate_id_to_img_id[cate_id].append(img_id)

print(from_cate_id_to_img_id[5])