# this script aims to generate the t-sne map for the gt bbox feature
# the previos step is to use the aggregate_gt_feature.py to collect the feature

import numpy as np
from sklearn.manifold import TSNE
import torch
import json

# load the collected image feature
#image_feat_path = '/home/zhuoming/finetuned_collect_result.json'
image_feat_path = '/home/zhuoming/raw_collect_result.json'
image_feat_content = json.load(open(image_feat_path))

all_cate_id = []
all_vectors = []
# handle the phrase
for cate_id in image_feat_content['from_catid_to_feat']:
    all_cate_id.append(cate_id)
    all_vectors.append(torch.tensor(image_feat_content['from_catid_to_feat'][cate_id]))


# load the text embedding, obtain the order
#text_embeddings = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_65cates.pt')
text_embeddings = torch.load('/data/zhuoming/detection/embeddings/raw_65cates.pt')

cate_name_65 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
 'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
 'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
 'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']
from_catid_to_cate_name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
from_cate_name_to_catid = {from_catid_to_cate_name[key]:key for key in from_catid_to_cate_name}
cate_id_65 = [from_cate_name_to_catid[name] for name in cate_name_65]

# note the mapping from the name to the idx
from_cate_id_to_idx_in_embed = {cate_id:i for i, cate_id in enumerate(cate_id_65)}

# append the text embeddings
all_vectors.append(text_embeddings)
all_vectors = torch.cat(all_vectors, dim=0)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_vectors)

print(X_embedded.shape)

all_result = {'all_tsne_xy': X_embedded.tolist(), 'all_cate_id': all_cate_id, 
              'from_catid_to_name': image_feat_content['from_catid_to_name'],
              'from_cate_id_to_idx_in_embed': from_cate_id_to_idx_in_embed}




#file = open('finetuned_collect_result_after_tsne.json', 'w')
file = open('raw_collect_result_after_tsne.json', 'w')
file.write(json.dumps(all_result))
file.close()
