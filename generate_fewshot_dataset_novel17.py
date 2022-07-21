# this script for validate the novel20 fewshot dataset and
# generate the novel17 dataset for the zero-shot setting
import json
from random import sample

# novel20_file = "/data/zhuoming/detection/few_shot_ann/coco/attention_rpn_10shot/official_10_shot_from_instances_train2017.json"
# novel20_content = json.load(open(novel20_file))

# # target 391895
# train_file = "/data/zhuoming/detection/coco/annotations/instances_train2017.json"
# val_file = "/data/zhuoming/detection/coco/annotations/instances_val2017.json"

# train_content = json.load(open(train_file))
# val_content = json.load(open(val_file))

# for ele in train_content['images']:
#     if ele['id'] == 391895:
#         print('get')
#         break
    
# for ele in val_content['images']:
#     if ele['id'] == 391895:
#         print('get')
#         break

# # get all the image id 
# all_img_id = []
# for ele in novel20_content['annotations']:
#     all_img_id.append(ele['image_id'])

# len(all_img_id)
# len(list(set(all_img_id)))
# # expriments show the image is select from the training set



# in the following of this script we will randomly sample the annotation
# for the novel categories for each image just sample one instance
train_file = "/data/zhuoming/detection/coco/annotations/instances_train2017.json"
train_content = json.load(open(train_file))
num_of_shot = 50

# obtain the novel cate ids
novel17_cate_name = ['airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors']

from_name_to_cate_id = {ele['name']:ele['id'] for ele in train_content['categories']}
novel17_cate_id = [from_name_to_cate_id[name] for name in novel17_cate_name]


from_cate_id_to_anno = {}
# aggregate the annotations base on the categories
for ele in train_content['annotations']:
    cate_id = ele['category_id']
    if cate_id not in from_cate_id_to_anno:
        from_cate_id_to_anno[cate_id] = []
    from_cate_id_to_anno[cate_id].append(ele)  

sample_images = []
sample_anno = []
# select 10 shot for each novel categories
for cate_id in novel17_cate_id:
    all_anno_for_cate = from_cate_id_to_anno[cate_id]
    sampled_anno_count = 0
    for anno in all_anno_for_cate:
        image_id = anno['image_id']
        #iscrowd = anno['iscrowd']
        bbox = anno['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 1025:
            continue
        # if the image has been selected, skip this image
        if image_id in sample_images:
            continue
        print(area, anno)
        sample_images.append(image_id)
        sample_anno.append(anno)
        sampled_anno_count += 1
        if sampled_anno_count >= num_of_shot:
            break

print(len(sample_images))
print(len(sample_anno))

final_json_content = {'info':train_content['info'], 'licenses':train_content['licenses'], 
                      'images':train_content['images'], 'annotations':sample_anno, 
                      'categories':train_content['categories']}
save_path = "/data/zhuoming/detection/few_shot_ann/coco/attention_rpn_10shot/novel17_10_shot_from_instances_train2017.json"

file = open(save_path, 'w')
file.write(json.dumps(final_json_content))
file.close()
