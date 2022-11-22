# this script aim to aggregate the GT feature of both base and novel categories
# for each base and novel categories we collect 20 gt feature for visualization

# the next step is to use the script tsne_test_1.py to generate the t-snt embedding.

from os import listdir
import os
import json
import random

# obtain the mapping the gt id to the coco_cate_id
from_catid_to_gtid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
from_gtid_to_catid = {from_catid_to_gtid[key] : key for key in from_catid_to_gtid}
# obtain all the novel id and base id
from_catid_to_cate_name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
from_cate_name_to_catid = {from_catid_to_cate_name[key]:key for key in from_catid_to_cate_name}
base_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

novel_cate_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')
base_cate_id = [from_cate_name_to_catid[ele] for ele in base_cate_name]
novel_cate_id = [from_cate_name_to_catid[ele] for ele in novel_cate_name]

# obtain all the file under the path
#gt_feat_path = '/data/zhuoming/detection/coco/clip_proposal_feat/base48_finetuned_base_filtered/gt'
gt_feat_path = '/project/nevatia_174/zhuoming/detection/coco/clip_proposal_feat/raw/gt'

onlyfiles = [f for f in listdir(gt_feat_path) if os.path.isfile(os.path.join(gt_feat_path, f))]


from_catid_to_feat = {catid:[] for catid in (base_cate_id + novel_cate_id)}
# iterate all the files to collect the feature
for file_name in onlyfiles:
    file_path = os.path.join(gt_feat_path, file_name)
    # load the content, collcted the feat to respective file
    file_content = json.load(open(file_path))
    all_feats = file_content['feat']
    all_gt_labels = file_content['gt_labels']
    for feat, gtid in zip(all_feats, all_gt_labels):
        catid = from_gtid_to_catid[gtid]
        if catid not in from_catid_to_feat:
            continue
        from_catid_to_feat[catid].append(feat)
    # check whether the collection is finish, other wise continue
    finish = True
    for catid in from_catid_to_feat:
        if len(from_catid_to_feat[catid]) < 20:
            finish = False
            continue
    if finish:
        break

# remove the duplicated feature for each categories, maintain only 10 feature each file
final_result = {}
for key in from_catid_to_feat:
    all_feats = from_catid_to_feat[key]
    random.shuffle(all_feats)
    final_result[key] = all_feats[:20]

result = {'from_catid_to_feat': final_result, 'from_catid_to_name':from_catid_to_cate_name}
file = open('collect_result.json', 'w')
file.write(json.dumps(result))
file.close()