import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os

# read the result
file_name = 'C:\\Users\\Zhuoming Liu\\Desktop\\text_embedding_tsne.json'
file_content = json.load(open(file_name))

all_x_list = []
all_y_list = []
for key in file_content:
    tsne_res = file_content[key]
    all_x_list.append(tsne_res[0])
    all_y_list.append(tsne_res[1])

all_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

all_cate_x_list = []
all_cate_y_list = []
for name in all_cate_name:
    tsne_res = file_content[name]
    all_cate_x_list.append(tsne_res[0])
    all_cate_y_list.append(tsne_res[1])


root = 'C:\\Users\\Zhuoming Liu\\Desktop\\embedding_tsne'
for name in all_cate_name:
    plt.axis('off')
    ax = plt.gca()
    #设置x轴、y轴名称
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    scat = ax.scatter(all_x_list,all_y_list, c='black', s=1)
    tsne_res = file_content[name]
    scat = ax.scatter(tsne_res[0], tsne_res[1], c='red', s=3)
    #plt.show()
    #file_name = name + '_vs_all_phrases.pdf'
    file_name = name + '_vs_all_phrases.png'
    plt.title(name + ' vs all phrases')
    plt.savefig(os.path.join(root, file_name))
    plt.close()