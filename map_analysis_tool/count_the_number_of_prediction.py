# this script aims to remove count the number of predition of
# of each category
import json
from unicodedata import category
from collections import OrderedDict

#pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10/base_results_65cates.bbox.json'))

#pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48_even_no_sigmoid/novel_results_65cates.bbox.json'))
#pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48_even/novel_results_65cates.bbox.json'))
#pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48_no_sigmoid/novel_results_65cates.bbox.json'))
pred_res = json.load(open('/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48/novel_results_65cates.bbox.json'))


from_cate_id_to_count = {}

for pred in pred_res:
    category_id = pred['category_id']
    if category_id not in from_cate_id_to_count:
        from_cate_id_to_count[category_id] = 0
    from_cate_id_to_count[category_id] += 1

print(from_cate_id_to_count)


novel_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

base_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')


gt_file = json.load(open('/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'))

from_name_to_id = {}
for anno in gt_file['categories']:
    name = anno['name']
    cate_id = anno['id']
    from_name_to_id[name] = cate_id

novel_count = sum([from_cate_id_to_count[from_name_to_id[name]] for name in novel_name if from_name_to_id[name] in from_cate_id_to_count])
base_count = sum([from_cate_id_to_count[from_name_to_id[name]] for name in base_name if from_name_to_id[name] in from_cate_id_to_count])    