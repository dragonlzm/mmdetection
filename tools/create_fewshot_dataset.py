import json

original_anno = json.load(open('C:\\Users\\XPS\Desktop\\annotations\\instances_train2017.json'))


NOVEL_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
  'cow', 'bottle', 'chair', 'couch', 'potted plant',
                'dining table', 'tv')

from_clsname_to_clsid = {item['name']:item['id'] for item in original_anno['categories']}
from_imagid_to_imginfo = {item['id']:item for item in original_anno['images']}

selected_image_id = {}
selected_image_info = []
selected_anno_info = []

# for each class we select 20 bboxes annotations.
# and it related image will be selected.
for cls_name in NOVEL_CLASSES:
    target_cls_id = from_clsname_to_clsid[cls_name]
    img_count = 0
    #anno_count = 0
    #while img_count < 10:
    for anno in original_anno['annotations']:
        anno_cls = anno['category_id']
        anno_img_id = anno['image_id']
        if anno['iscrowd'] == 1:
            continue
        if anno_img_id in selected_image_id:
            continue
        if anno_cls != target_cls_id:
            continue
        # the unselect image and it has an annotation for the target task.
        selected_image_id[anno_img_id] = 1
        selected_image_info.append(from_imagid_to_imginfo[anno_img_id])
        selected_anno_info.append(anno)
        img_count += 1
        if img_count >= 10:
            break

result = {}
result['info'] = original_anno['info']
result['licenses'] = original_anno['licenses']
result['images'] = selected_image_info
result['annotations'] = selected_anno_info
result['categories'] = original_anno['categories']

with open('fewshot.json', 'w') as outfile:
    json.dump(result, outfile)

