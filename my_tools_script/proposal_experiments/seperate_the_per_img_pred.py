# this script aims to seperate the prediction into per image result
import json
import os

file_root = '/data/zhuoming/detection/coco/vitdet_proposal_val'
# load the full prediction
full_pred_path = '/data/zhuoming/detection/grad_clip_check/proposal_selector_v2_coco.gt_acc.json'
full_pred_content = json.load(open(full_pred_path))

for ele in full_pred_content:
    image_id = ele['image_id']
    save_file_name = str(image_id) + '_module_score.json'
    save_file_content = {'module_obj_score': ele['score']}
    file = open(os.path.join(file_root, save_file_name), 'w')
    file.write(json.dumps(save_file_content))
    file.close()
