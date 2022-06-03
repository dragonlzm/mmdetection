# this script aims to calculate the cosine similarity between the feat of mask rcnn and the feat of the finetuned clip
import os
import json
import torch

# load the original bbox
bbox_path_root = '/home/zhuoming/base_novel_clippro'
bbox_files = [f for f in os.listdir(bbox_path_root) if os.path.isfile(os.path.join(bbox_path_root, f))]

# the path for mask rcnn feat
mask_rcnn_feat_root = '/home/zhuoming/mask_rcnn_feat'
# the path for clip feat
clip_feat_root = '/home/zhuoming/clip_feat/random'

cos_base = 0
all_base_num = 0
cos_novel = 0
all_novel_num = 0
cos_trained_clip = 0
all_trained_clip_num = 0
cos_untrained_clip = 0
all_untrained_clip_num = 0

for file in bbox_files:
    bbox_file_content = json.load(open(os.path.join(bbox_path_root, file)))
    mask_rcnn_feat = torch.tensor(json.load(open(os.path.join(mask_rcnn_feat_root, file)))['feat'])
    clip_feat = torch.tensor(json.load(open(os.path.join(clip_feat_root, file)))['feat'])
    base_num = len(bbox_file_content['base'])
    novel_num = len(bbox_file_content['novel'])
    clip_num = len(bbox_file_content['clip'])
    # calculate the average cosine similarity of base gt
    if base_num != 0:
        all_base_num += base_num
        # obtain the feat
        mask_rcnn_base_feat = mask_rcnn_feat[:base_num]
        clip_base_feat = clip_feat[:base_num]
        # nomalize the feat
        mask_rcnn_base_feat = mask_rcnn_base_feat / mask_rcnn_base_feat.norm(dim=-1, keepdim=True)
        clip_base_feat = clip_base_feat / clip_base_feat.norm(dim=-1, keepdim=True)
        # calculate the cos similarity
        base_cos_value = (mask_rcnn_base_feat * clip_base_feat).sum().item()
        cos_base += base_cos_value
    
    # calculate the average cosine similarity of novel gt 
    if novel_num != 0:
        all_novel_num += novel_num
        # obtain the feat
        mask_rcnn_novel_feat = mask_rcnn_feat[base_num:base_num+novel_num]
        clip_novel_feat = clip_feat[base_num:base_num+novel_num]
        # normalize the feat
        mask_rcnn_novel_feat = mask_rcnn_novel_feat / mask_rcnn_novel_feat.norm(dim=-1, keepdim=True)
        clip_novel_feat = clip_novel_feat / clip_novel_feat.norm(dim=-1, keepdim=True)
        # calculate the cos similarity
        novel_cos_value = (mask_rcnn_novel_feat * clip_novel_feat).sum().item()
        cos_novel += novel_cos_value
        
    # calculate the average cosine similarity of trained distillation feat
    all_trained_clip_num += 200
    # obtain the feat
    mask_rcnn_trained_clip_feat = mask_rcnn_feat[base_num+novel_num:base_num+novel_num+200]
    clip_trained_clip_feat = clip_feat[base_num+novel_num:base_num+novel_num+200]
    # normalize the feat
    mask_rcnn_trained_clip_feat = mask_rcnn_trained_clip_feat / mask_rcnn_trained_clip_feat.norm(dim=-1, keepdim=True)
    clip_trained_clip_feat = clip_trained_clip_feat / clip_trained_clip_feat.norm(dim=-1, keepdim=True)
    # calculate the cos similarity
    trained_clip_cos_value = (mask_rcnn_trained_clip_feat * clip_trained_clip_feat).sum().item()
    cos_trained_clip += trained_clip_cos_value
    
    # calculate the average cosine similarity of untrained distillation feat
    all_untrained_clip_num += 200
    # obtain the feat
    mask_rcnn_untrained_clip_feat = mask_rcnn_feat[base_num+novel_num+200:base_num+novel_num+400]
    clip_untrained_clip_feat = clip_feat[base_num+novel_num+200:base_num+novel_num+400]
    # normalize the feat
    mask_rcnn_untrained_clip_feat = mask_rcnn_untrained_clip_feat / mask_rcnn_untrained_clip_feat.norm(dim=-1, keepdim=True)
    clip_untrained_clip_feat = clip_untrained_clip_feat / clip_untrained_clip_feat.norm(dim=-1, keepdim=True)
    # calculate the cos similarity
    untrained_clip_cos_value = (mask_rcnn_untrained_clip_feat * clip_untrained_clip_feat).sum().item()
    cos_untrained_clip += untrained_clip_cos_value
    
print('base gt feat:', cos_base/all_base_num, 'novel gt feat', cos_novel/all_novel_num, 'trained clip feat', cos_trained_clip/all_trained_clip_num, 'untrained clip feat', cos_untrained_clip/all_untrained_clip_num)  