import json
import os
import torch

file_root = '/home/zhuoming/detectron_proposal2'

all_final_name = os.listdir(file_root)
# load the final prediction(obtain the bboxes)
all_proposal_files = [ele for ele in all_final_name if ele.endswith('.jpg.json')]
# all_final_pred_files = [ele for ele in all_final_name if ele.endswith('.jpg_final_pred.json')]
# all_clip_pred_files = [ele for ele in all_final_name if ele.endswith('.jpg_clip_pred.json')]

# prepare which idx is the novel
all_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']
novel_name = ['airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 
            'tie', 'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 
            'keyboard', 'sink', 'scissors']
from_name_to_idx = {name:i for i, name in enumerate(all_names)}
novel_name_idx = [from_name_to_idx[name] for name in novel_name]

# print(len(all_proposal_files), len(all_final_pred_files), len(all_clip_pred_files))
for proposal_file in all_proposal_files:
    final_pred_files = proposal_file[:-5] + '_final_pred.json'
    clip_pred_files = proposal_file[:-5] + '_clip_pred.json'
    replaced_file_name = proposal_file[:-5] + '_final_replaced.json'
    # if os.path.exists(os.path.join(file_root, replaced_file_name)):
    #     continue
    # load the rpn prediction(obtain the objectness score)
    proposal_file_content = json.load(open(os.path.join(file_root, proposal_file)))
    objectness_score = torch.tensor(proposal_file_content['score'])
    # load the clip prediction score
    clip_predicted_content = json.load(open(os.path.join(file_root, clip_pred_files)))
    clip_predicted_score = torch.tensor(clip_predicted_content['score'])
    # select the bbox which maxscore of the clip score is novel categories
    value, max_cate_idx = torch.max(clip_predicted_score, dim=-1)
    novel_idx = torch.tensor([i for i, ele in enumerate(max_cate_idx) if ele in novel_name_idx])
    if len(novel_idx) == 0:
        needed_objectness_score = None
        clip_confidence_score = None
    else:
        needed_objectness_score = objectness_score[novel_idx].tolist()
        clip_confidence_score = value[novel_idx].tolist()
    
    # dump the result
    final_pred_files = proposal_file[:-5] + '_final_pred.json'
    final_pred_content = json.load(open(os.path.join(file_root, final_pred_files)))
    final_pred_content['novel_idx'] = novel_idx.tolist()
    final_pred_content['objectness_score'] = needed_objectness_score
    
    file = open(os.path.join(file_root, replaced_file_name), 'w')
    file.write(json.dumps(final_pred_content))
    file.close()
    