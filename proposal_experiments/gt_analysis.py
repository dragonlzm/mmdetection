import os 
import json
import torch

path = "/home/zhuoming/detectron_proposal2"
file_list = os.listdir(path)

filtered_file_list = [ele for ele in file_list if "match_gt" in ele]

mis_cls_conf = []
corr_cls_conf = []

for file_name in filtered_file_list:
    # load the file
    file_content = json.load(open(file_name))
    # load the gt index
    gt_idxs = torch.tensor(file_content['all_target_proposal_gt_idx'])
    if len(gt_idxs) == 0:
        continue
    # load the clip distribution
    clip_distri = torch.tensor(file_content['all_target_proposal_clip_distri'])
    # obtain the max score and the idx
    clip_confs, clip_idxs = torch.max(clip_distri, dim=-1)
    # compare whether the idx is matched
    for _gt_label, _clip_label, _clip_conf in zip(gt_idxs, clip_idxs, clip_confs):
        if _gt_label == _clip_label:
            corr_cls_conf.append(_clip_conf.unsqueeze(dim=0))
        else:
            mis_cls_conf.append(_clip_conf.unsqueeze(dim=0))

print('mean correct score', torch.mean(torch.cat(corr_cls_conf)), 'mean mis cls score', torch.mean(torch.cat(mis_cls_conf)))

corr_cls_conf = torch.cat(corr_cls_conf).tolist()
mis_cls_conf = torch.cat(mis_cls_conf).tolist()

final_pred_content = {"corr_cls_conf":corr_cls_conf , "mis_cls_conf":mis_cls_conf}

file = open(os.path.join(path, "confidence.json"), 'w')
file.write(json.dumps(final_pred_content))
file.close()