import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# read the result

#tsne_res = np.load('tsne.npy')

#assign_res = np.load('assigned_res.npy')
assign_res = np.load('assigned_res_4_by_4.npy')
#assign_res = np.load('assigned_gt_res.npy')
#assign_res = np.load('assigned_gt_res_new.npy')
print(assign_res.shape)

count = 0
from_ori_id_to_new_id = {}
new_assign_id = np.zeros(assign_res.shape)
for i in range(assign_res.shape[0]):
    old_id = assign_res[i]
    if old_id not in from_ori_id_to_new_id:
        from_ori_id_to_new_id[old_id] = count
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
#tsne_res = np.load('tsne_gt_new.npy')

tsne_res = np.load('tsne_fasterrcnn_4_by_4_final_feat.npy')
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
plt.axis('off')
ax = plt.gca()
#设置x轴、y轴名称
ax.set_xlabel('x')
ax.set_ylabel('y')

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
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

'''
# filter the bg
filter_x_list = []
filter_y_list = []
filtered_new_assign_id = []


for x, y, assign_id in zip(x_list, y_list, new_assign_id):
    #if assign_id == 2:
    if assign_id == 0:
        continue
    filter_x_list.append(x)
    filter_y_list.append(y)
    filtered_new_assign_id.append(assign_id)'''


# only bg
filter_x_list = []
filter_y_list = []
filtered_new_assign_id = []

for x, y, assign_id in zip(x_list, y_list, new_assign_id):
    if assign_id != 0:
        continue
    filter_x_list.append(x)
    filter_y_list.append(y)
    filtered_new_assign_id.append(assign_id)


'''
# filter the bg
filter_x_list = []
filter_y_list = []
filtered_new_assign_id = []

for x, y, assign_id in zip(x_list, y_list, new_assign_id):
    if assign_id == 2:
        filter_x_list.append(x)
        filter_y_list.append(y)
        filtered_new_assign_id.append(0)
    if assign_id == 28:
        filter_x_list.append(x)
        filter_y_list.append(y)
        filtered_new_assign_id.append(1)'''

# make the scatter
#scat = ax.scatter(x_list,y_list,c=new_assign_id, s=1, cmap=cmap, norm=norm)

scat = ax.scatter(filter_x_list, filter_y_list, c=filtered_new_assign_id, s=1,  cmap=cmap, norm=norm)

ax.scatter([x_list[81], x_list[82], x_list[102]], [y_list[81], y_list[82], y_list[102]], c='red', s=5, alpha=0.5)

#ax.scatter(x_list, y_list, c='black', s=2, alpha=0.5)

cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
cb.set_label('Custom cbar')
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
ax.set_title('fasterrcnn patch feature tsne visualization(bg only)')
#ax.set_title('fasterrcnn patch feature tsne visualization(fg only)')
plt.show()