# this script aims to generate the subset for the lvis dataset 
# will make use of all novel categories, for each categories will select one image
# also select 100 image which do not have the novel categories
import json
json_path = "/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_train.json"
json_content = json.load(open(json_path))

# aggregate annotation information for each categories
from_cate_id_to_image_id = {}
for anno in json_content['annotations']:
    category_id = anno['category_id']
    image_id = anno['image_id']
    if category_id not in from_cate_id_to_image_id:
        from_cate_id_to_image_id[category_id] = []
    from_cate_id_to_image_id[category_id].append(image_id)

# make the image id unique
for category_id in from_cate_id_to_image_id:
    from_cate_id_to_image_id[category_id] = list(set(category_id))

# get the categories id
novel_id = []
base_id = []
for ele in json_content['categories']:
    frequency = ele['frequency']
    category_id = ele['id']
    if frequency == 'c' or frequency == 'f':
        base_id.append(category_id)
    else:
        novel_id.append(category_id)

print(len(novel_id), len(base_id))

# for each novel categories, get one image
all_selected_image_id = []
for category_id in novel_id:
    all_image_id_for_now_cate = from_cate_id_to_image_id[category_id]
    for image_id in all_image_id_for_now_cate:
        if image_id not in all_selected_image_id:
            all_selected_image_id.append(image_id)
            break
        else:
            continue

# select 100 base image
for i, category_id in enumerate(base_id):
    all_image_id_for_now_cate = from_cate_id_to_image_id[category_id]
    for image_id in all_image_id_for_now_cate:
        if image_id not in all_selected_image_id:
            all_selected_image_id.append(image_id)
            break
        else:
            continue
    
    if i > 100:
        break

# collect the image info
from_image_id_to_image_info = {}
for ele in json_content['images']:
    image_id = ele['id']
    from_image_id_to_image_info[image_id] = ele

# collect the annotation info
from_image_id_to_anno_info = {}
for ele in json_content['annotations']:
    image_id = ele['image_id']
    if image_id not in from_image_id_to_anno_info:
        from_image_id_to_anno_info[image_id] = []
    from_image_id_to_anno_info[image_id].append(ele)

all_image_info = []
all_anno_info = []
for image_id in all_selected_image_id:
    all_image_info.append(from_image_id_to_image_info[image_id])
    all_anno_info += from_image_id_to_anno_info[image_id]

all_result = {'info':json_content['info'], 
              'annotations':all_anno_info, 
              'images':all_image_info, 
              'licenses':json_content['licenses'], 
              'categories':json_content['categories']}

file = open('lvis_train_novel_base_subset.json', 'w')
file.write(json.dumps(all_result))
file.close()