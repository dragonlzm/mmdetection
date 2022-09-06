import json
import os

#gt_annotation_root = "/data/zhuoming/detection/coco/annotations/"
#gt_annotation_root = "/project/nevatia_174/zhuoming/detection/coco/annotations/"
#gt_annotation_root = "/data/zhuoming/detection/lvis_v1/annotations/"
gt_annotation_root = "/project/nevatia_174/zhuoming/detection/lvis_v1/annotations/"
#gt_annotation_file = "instances_train2017.json"
gt_annotation_file = "lvis_v1_train.json"
gt_annotation_path = os.path.join(gt_annotation_root, gt_annotation_file)
full_annotation_file = json.load(open(gt_annotation_path))

# create mapping from image_id to image info
image_id_list = []
from_image_id_to_image_info = {}
for image_info in full_annotation_file['images']:
    image_id = image_info['id']
    image_id_list.append(image_id)
    from_image_id_to_image_info[image_id] = image_info


# create mapping from image_id to anno info
from_image_id_to_annotation = {}
for anno in full_annotation_file['annotations']:
    image_id = anno['image_id']
    if image_id not in from_image_id_to_annotation:
        from_image_id_to_annotation[image_id] = []
    from_image_id_to_annotation[image_id].append(anno)

# split the data set
# for coco
#split_num = 15
# for lvis 
split_num = 13

split_idx = [0] + [(i+1)*8000 for i in range(split_num)]
print(split_idx)

# save each part of the data
for i in range(len(split_idx) - 1):
    start_idx = split_idx[i]
    end_idx = split_idx[i+1]
    # get the idx list 
    now_img_id_list = image_id_list[start_idx:end_idx] 
    # get the image_info list 
    now_image_info_list = []
    for image_id in now_img_id_list:
        now_image_info_list.append(from_image_id_to_image_info[image_id])
    # get the annotation list
    now_all_anno = []
    for image_id in now_img_id_list:
        if image_id not in from_image_id_to_annotation:
            continue
        now_all_anno += from_image_id_to_annotation[image_id]
    result_json = {'info':full_annotation_file['info'], 'licenses':full_annotation_file['licenses'], 
                    'images':now_image_info_list, 'annotations':now_all_anno, 'categories':full_annotation_file['categories']}
    # for coco
    #file_name = "instances_train2017_" + str(start_idx) + '_' + str(end_idx) + ".json"
    # for lvis
    file_name = "lvis_v1_train_" + str(start_idx) + '_' + str(end_idx) + ".json"
    file_name = os.path.join(gt_annotation_root, file_name)
    file = open(file_name, 'w')
    file.write(json.dumps(result_json))
    file.close()


