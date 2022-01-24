import numpy as np
from sklearn.manifold import TSNE
import torch
#from tsnecuda import TSNE
#X_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)

#ori_feat = torch.load('feature.pt').reshape([-1, 512]).numpy()
#ori_feat = torch.load('feature_4_by_4.pt').reshape([-1, 512]).numpy()
#ori_feat = torch.load('fasterrcnn_final_feat_gt.pt').numpy()
#ori_feat = torch.load('fasterrcnn_4_by_4_final_feat.pt').numpy()
#ori_feat = torch.load('gt_feature.pt').numpy()

ori_feat = torch.load('feature_4_by_4.pt').reshape([-1, 512])
feature_4_by_4_fliped = torch.load('000000281693_feature_4_by_4_fliped.pt').reshape([-1, 512])
feature_4_by_4 = torch.load('000000281693_feature_4_by_4.pt').reshape([-1, 512])
feature_8_by_8_fliped = torch.load('000000281693_feature_8_by_8_fliped.pt').reshape([-1, 512])
feature_8_by_8 = torch.load('000000281693_feature_8_by_8.pt').reshape([-1, 512])

whole_img_feat = torch.load('000000281693_whole_img_feat.pt').reshape([-1, 512]).cpu()

feat_list = [ori_feat, feature_4_by_4_fliped, feature_4_by_4, feature_8_by_8_fliped, feature_8_by_8, whole_img_feat]

all_feat = torch.cat(feat_list, dim=0)
#temp_origin_feat = ori_feat[:64000]

#X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(temp_origin_feat)
#X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(ori_feat)
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(all_feat)


print(X_embedded.shape)

#np.save('tsne.npy', X_embedded)

#np.save('tsne_all.npy', X_embedded)

#np.save('tsne_gt_new.npy', X_embedded)

#np.save('tsne_fasterrcnn_final_feat_gt.npy', X_embedded)

#np.save('tsne_fasterrcnn_4_by_4_final_feat.npy', X_embedded)

np.save('tsne_airplane_with_4_by_4.npy', X_embedded)
