# this script aims to use the CLIP feat for classification, to see the feat is correct or not
# this script aims to calculate the cosine similarity between the feat of mask rcnn and the feat of the finetuned clip
import os
import json
import torch

# load the original bbox
bbox_path_root = '/home/zhuoming/base_novel_clippro'


# the path for clip feat
#root = '/home/zhuoming/clip_feat/random'
#feat_root = '/home/zhuoming/mask_rcnn_feat'
#feat_root = '/home/zhuoming/mask_rcnn_feat_2x'

#feat_root = '/home/zhuoming/clip_feat_all/random'
#feat_root = '/home/zhuoming/mask_rcnn_feat_all'
feat_root = '/home/zhuoming/mask_rcnn_feat_2x_all'

all_files = [f for f in os.listdir(feat_root) if os.path.isfile(os.path.join(feat_root, f))]

# load the embedding
base_embeddings = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_48cates.pt')
novel_embeddings = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_17cates.pt')

base_embeddings = base_embeddings / base_embeddings.norm(dim=-1, keepdim=True)
novel_embeddings = novel_embeddings / novel_embeddings.norm(dim=-1, keepdim=True)

# from cate_name to idx
base_cates_name = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')
novel_cates_name = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')
from_name_to_idx_base = {name:i for i, name in enumerate(base_cates_name)}
from_name_to_idx_novel = {name:i for i, name in enumerate(novel_cates_name)}

# from cate_id to cate_name 
gt_file_content = json.load(open('/data/zhuoming/detection/coco/annotations/train_100imgs.json'))
from_cate_id_to_cate_name = {ele['id']:ele['name'] for ele in gt_file_content['categories']}

all_base_num = 0
all_base_match_num = 0
all_novel_num = 0
all_novel_match_num = 0

for i, file in enumerate(all_files):
    bbox_file_content = json.load(open(os.path.join(bbox_path_root, file)))
    clip_feat = torch.tensor(json.load(open(os.path.join(feat_root, file)))['feat'])
    # seperate the novel and the base 
    base_bbox_num = len(bbox_file_content['base'])
    novel_bbox_num = len(bbox_file_content['novel'])
    if base_bbox_num != 0:
        all_base_num += base_bbox_num
        # obtain the feat
        clip_base_feat = clip_feat[:base_bbox_num]
        # nomalize the feat
        cls_score = torch.matmul(clip_base_feat, base_embeddings.t())
        # calculate the cos similarity
        max_score, max_idx = torch.max(cls_score, dim=1)
        
        # convert the cate_id to idx
        all_base_cate_id = bbox_file_content['base_cates']
        all_base_cate_idx = torch.tensor([from_name_to_idx_base[from_cate_id_to_cate_name[id]] for id in all_base_cate_id])
        
        all_base_match_num += ((all_base_cate_idx == max_idx).sum())
    if novel_bbox_num != 0:
        all_novel_num += novel_bbox_num
        # obtain the feat
        clip_novel_feat = clip_feat[base_bbox_num:base_bbox_num+novel_bbox_num]
        # nomalize the feat
        cls_score = torch.matmul(clip_novel_feat, novel_embeddings.t())
        # calculate the cos similarity
        max_score, max_idx = torch.max(cls_score, dim=1)
        
        # convert the cate_id to idx
        all_novel_cate_id = bbox_file_content['novel_cates']
        all_novel_cate_idx = torch.tensor([from_name_to_idx_novel[from_cate_id_to_cate_name[id]] for id in all_novel_cate_id])
        
        all_novel_match_num += ((all_novel_cate_idx == max_idx).sum())
    if i % 100 == 0:
        print(i)
        
print(all_base_match_num / all_base_num, all_novel_match_num / all_novel_num)