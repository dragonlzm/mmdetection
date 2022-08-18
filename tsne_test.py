# this script aims to generate the t-sne map

import numpy as np
from sklearn.manifold import TSNE
import torch
import json

phrase_file_path = '/data/zhuoming/detection/coco/annotations/phrases_vs_feats.json'
phrase_content = json.load(open(phrase_file_path))

all_cate_file_path = '/data/zhuoming/detection/coco/annotations/coco_names_vs_feats.json'
all_cate_content = json.load(open(all_cate_file_path))

all_names = []
all_vectors = []
# handle the phrase
for key in phrase_content:
    all_names.append(key)
    all_vectors.append(torch.tensor(phrase_content[key]))

# handle the categories name:
for key in all_cate_content:
    if key in all_names:
        continue
    all_names.append(key)
    all_vectors.append(torch.tensor(all_cate_content[key]))

all_vectors = torch.cat(all_vectors, dim=0)

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_vectors)

print(X_embedded.shape)

all_result = {}
for name, pos in zip(all_names, X_embedded):
    all_result[name] = pos.tolist()

file = open('text_embedding_tsne.json', 'w')
file.write(json.dumps(all_result))
file.close()
