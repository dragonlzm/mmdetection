# this script is to visualize the t-sne embedding with only two color, the novel and the base will use different color
# the previous step is to use the to generate the feature tsne_test_1.py

import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import torch

# read the result
file_name = 'C:\\Users\\Zhuoming Liu\\Desktop\\finetuned_collect_result_after_tsne.json'
#file_name = 'C:\\Users\\Zhuoming Liu\\Desktop\\raw_collect_result_after_tsne.json'
file_content = json.load(open(file_name))

# all_result = {'all_tsne_xy': X_embedded.tolist(), 'all_cate_id': all_cate_id, 
#               'from_catid_to_name': image_feat_content['from_catid_to_name'],
#               'from_cate_id_to_idx_in_embed': from_cate_id_to_idx_in_embed}

# sperate the image feat and the text embedding
image_feat = torch.tensor(file_content['all_tsne_xy'][:-65])
text_embedding = torch.tensor(file_content['all_tsne_xy'][-65:])

# print('image_feat', image_feat.shape)
# print('text_embedding', text_embedding.shape)

base_cate_id = [1, 2, 3, 4, 7, 8, 9, 15, 16, 19, 20, 23, 24, 25, 27, 31, 33, 34, 35, 38, 42, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 62, 65, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84, 85, 86, 90]
novel_cate_id = [5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]

# reshape the point into(65, 20, 2)
image_feat = image_feat.reshape(65, 20, 2)

all_novel_x_list = []
all_novel_y_list = []

all_base_x_list = []
all_base_y_list = []
# iterate all the point and the cateid
for image_feat_per_cate, cate_id in zip(image_feat, file_content['all_cate_id']):
    cate_id = int(cate_id)
    #print('cate_id', cate_id in base_cate_id, cate_id, base_cate_id, type(cate_id))
    # if the category is base add to base list
    if cate_id in base_cate_id:
        all_x = image_feat_per_cate[:, 0].tolist()
        all_y = image_feat_per_cate[:, 1].tolist()
        all_base_x_list += all_x
        all_base_y_list += all_y
    # otherwise add to the base list
    else:
        all_x = image_feat_per_cate[:, 0].tolist()
        all_y = image_feat_per_cate[:, 1].tolist()
        all_novel_x_list += all_x
        all_novel_y_list += all_y


# aggargate the embeddings
all_novel_embedding_x_list = []
all_novel_embedding_y_list = []

all_base_embedding_x_list = []
all_base_embedding_y_list = []

from_cate_id_to_idx_in_embed = file_content['from_cate_id_to_idx_in_embed']
for cate_id in file_content['all_cate_id']:
    idx = from_cate_id_to_idx_in_embed[cate_id]
    now_embedding = text_embedding[idx].tolist()
    if int(cate_id) in base_cate_id:
        all_base_embedding_x_list.append(now_embedding[0])
        all_base_embedding_y_list.append(now_embedding[1])
    else:
        all_novel_embedding_x_list.append(now_embedding[0])
        all_novel_embedding_y_list.append(now_embedding[1])        


root = 'C:\\Users\\Zhuoming Liu\\Desktop\\'
plt.axis('off')
ax = plt.gca()
#设置x轴、y轴名称
ax.set_xlabel('x')
ax.set_ylabel('y')
scat = ax.scatter(all_base_x_list, all_base_y_list, c='black', s=1)
scat = ax.scatter(all_novel_x_list, all_novel_y_list, c='red', s=1)

scat = ax.scatter(all_base_embedding_x_list, all_base_embedding_y_list, c='black', s=5)
scat = ax.scatter(all_novel_embedding_x_list, all_novel_embedding_y_list, c='red', s=5) 
#plt.show()
file_name = 'Adapted CLIP features and embedding.pdf'
plt.title('Adapted CLIP features and embedding')

# file_name = 'Unadapted CLIP features and embedding.pdf'
# plt.title('Unadapted CLIP features and embedding')
plt.savefig(os.path.join(root, file_name))
plt.close()