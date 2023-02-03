# this script aims to convert the gt annotation into the format of 
# prediction
import json

ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'

gt_anno = json.load(open(ann_file))

# needed format
#{'image_id': 397133, 'bbox': [381.830322265625, 68.92253875732422, 120.26748657226562, 282.2848892211914], 'score': 0.9994545578956604, 'category_id': 1}
result_list = []
from_image_id_to_anno = {}
# {'segmentation': [[279.11, 370.25, 281.96, 214.56, 283.86, 209.81, 481.33, 215.51, 485.13, 370.25]], 'area': 31935.485049999996, 'iscrowd': 0, 'image_id': 464476, 'bbox': [279.11, 209.81, 206.02, 160.44], 'category_id': 72, 'id': 29131}
for ele in gt_anno['annotations']:
    image_id = ele['image_id']
    bbox = ele['bbox']
    category_id = ele['category_id']
    if image_id not in from_image_id_to_anno:
        from_image_id_to_anno[image_id] = []
    needed_info = {'image_id': image_id, 'bbox':bbox , 'score': 1.0, 'category_id': category_id}
    from_image_id_to_anno[image_id].append(needed_info)
    
for image_id in from_image_id_to_anno:
    all_anno = from_image_id_to_anno[image_id]
    packed_info = {'image_id': image_id, 'bbox':[1.0, 1.0, 1.0, 1.0] , 'score': 0.0, 'category_id': 1}
    while len(all_anno) < 100:
        all_anno.append(packed_info)
    from_image_id_to_anno[image_id] = all_anno
    
final_result = []

for image_id in from_image_id_to_anno:
    final_result += from_image_id_to_anno[image_id]

file = open('novel_gt_result.json', 'w')
file.write(json.dumps(final_result))
file.close()

    