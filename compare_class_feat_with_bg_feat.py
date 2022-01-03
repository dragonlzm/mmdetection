import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from_ori_id_to_cate_name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 
13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 
19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 
27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 
35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 
41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 
48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 
55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 
62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 
73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 
80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 
88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# read the result

#tsne_res = np.load('tsne.npy')

#assign_res = np.load('assigned_res.npy')
#assign_res = np.load('assigned_res_4_by_4.npy')
#assign_res = np.load('assigned_gt_res.npy')
assign_res = np.load('assigned_gt_res_new.npy')
print(assign_res.shape)

count = 0
from_ori_id_to_new_id = {}
from_new_id_to_ori_id = {}
new_assign_id = np.zeros(assign_res.shape)
for i in range(assign_res.shape[0]):
    old_id = assign_res[i]
    #if int(old_id) == 12:
    #    print('now is id 12')
    if old_id not in from_ori_id_to_new_id:
        from_ori_id_to_new_id[old_id] = count
        from_new_id_to_ori_id[count] = int(old_id)
        new_id = count
        count += 1
    else:
        new_id = from_ori_id_to_new_id[old_id]
    new_assign_id[i] = new_id


print(from_ori_id_to_new_id)
print(new_assign_id)

#tsne_res = np.load('tsne_all.npy')
#tsne_res = np.load('tsne_all_4_by_4.npy')
#tsne_res = np.load('tsne_gt.npy')
tsne_res = np.load('tsne_gt_new.npy')

#tsne_res = np.load('tsne_fasterrcnn_4_by_4_final_feat.npy')
#tsne_res = np.load('tsne_fasterrcnn_final_feat_gt.npy')

print(tsne_res.shape)

num, dim = tsne_res.shape

x_list = []
y_list = []
for i in range(num):
    x_list.append(tsne_res[i][0])
    y_list.append(tsne_res[i][1])

#创建图并命名
#plt.figure('Scatter fig')
#plt.figure(figsize=(6,6))


'''
#N = 1
#N = count
N = 2
# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)'''

# only bg
filter_x_list = []
filter_y_list = []
filtered_new_assign_id = []

for x, y, assign_id in zip(x_list, y_list, new_assign_id):
    #if assign_id != 0:
    if assign_id != 2:
        continue
    filter_x_list.append(x)
    filter_y_list.append(y)
    filtered_new_assign_id.append(assign_id)


print(from_ori_id_to_new_id)
for i in range(1, count):
    if i == 2:
        continue
    the_origin_id = from_new_id_to_ori_id[i]
    the_origin_name = from_ori_id_to_cate_name[the_origin_id]
    print(the_origin_id, the_origin_id, the_origin_name)
    plt.axis('off')
    ax = plt.gca()
    #设置x轴、y轴名称
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # filter the fg
    filter_fg_x_list = []
    filter_fg_y_list = []
    #filtered_new_assign_id = []

    for x, y, assign_id in zip(x_list, y_list, new_assign_id):
        if assign_id == i:
            filter_fg_x_list.append(x)
            filter_fg_y_list.append(y)

    # make the scatter
    #scat = ax.scatter(x_list,y_list,c=new_assign_id, s=1, cmap=cmap, norm=norm)

    #scat = ax.scatter(filter_x_list, filter_y_list, c=filtered_new_assign_id, s=1,  cmap=cmap, norm=norm)
    scat = ax.scatter(filter_x_list, filter_y_list, c='black', s=1, alpha=0.5)
    scat = ax.scatter(filter_fg_x_list, filter_fg_y_list, c='red', s=3, alpha=0.5)

    #ax.scatter([x_list[81], x_list[82], x_list[102]], [y_list[81], y_list[82], y_list[102]], c='red', s=5, alpha=0.5)

    #ax.scatter(x_list, y_list, c='black', s=2, alpha=0.5)

    #cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    #cb.set_label('Custom cbar')
    #ax.set_title('Discrete color mappings')
    #ax.set_title('GT feature tsne visualization')
    #ax.set_title('people vs tv features tsne visualization')
    #ax.set_title('gt features without bg tsne visualization')

    #ax.set_title('patch feature tsne visualization')

    #ax.set_title('patch feature(bg only) tsne visualization')
    #ax.set_title('patch feature(bg only) tsne visualization 4 by 4')
    #ax.set_title('patch feature(without bg) tsne visualization')
    #ax.set_title('patch feature(without bg) tsne visualization 4 by 4')
    #ax.set_title('fasterrcnn GT bbox feature tsne visualization')
    #ax.set_title('fasterrcnn patch feature tsne visualization')
    #ax.set_title('fasterrcnn patch feature tsne visualization(all colors)')
    #ax.set_title('fasterrcnn patch feature tsne visualization(bg only)')
    #ax.set_title('fasterrcnn patch feature tsne visualization(fg only)')
    #plt.show()
    #ax.set_title('patch feature(4 by 4) tsne visualization bg vs ' + the_origin_name)
    ax.set_title('gt feature tsne visualization person vs ' + the_origin_name)

    #plt.savefig('C:\\Users\\XPS\\Desktop\\experiment_result\\clip\\bg_contrast_with_each_class(4_by_4_and_patch)\\bg vs ' + the_origin_name + '.png',bbox_inches='tight',pad_inches=0.0)
    plt.savefig('C:\\Users\\XPS\\Desktop\\experiment_result\\clip\\gt_person_vs_others\\person vs ' + the_origin_name + '.png',bbox_inches='tight',pad_inches=0.0)

    
    plt.close()