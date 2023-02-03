# this script aims to combine the result of proposal selection and its categories.
import json
import os
import torch

# load the predicted iou
predicted_iou_file = '/data/zhuoming/detection/test/proposal_selector_coco_with_pre_nms_pred/testing.gt_acc.json'
# aggregate the result of each image
predicted_iou = json.load(open(predicted_iou_file))

from_image_id_to_predicted_score = {}
for res in predicted_iou:
    image_id = res['image_id']
    all_pred_score = res['score'][0]
    if image_id not in from_image_id_to_predicted_score:
        from_image_id_to_predicted_score[image_id] = all_pred_score

# obtain the file name:
gt_anno_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'
gt_res = json.load(open(gt_anno_file))
from_image_id_to_image_file_name = {}

# obtain the conversion from image id to file name
for ele in gt_res['images']:
    image_id = ele['id']
    file_name = ele['file_name']
    from_image_id_to_image_file_name[image_id] = file_name

all_final_prediction = []
prediction_root = '/data/zhuoming/detection/coco/bn65_val_prediction'
for image_id in from_image_id_to_predicted_score:
    predict_scores = torch.tensor(from_image_id_to_predicted_score[image_id])
    filename = from_image_id_to_image_file_name[image_id]
    full_file_path = os.path.join(prediction_root, '.'.join(filename.split('.')[:-1]) + '.json')
    # load the prediction bboxes and cate for each image
    pred_file_content = json.load(open(full_file_path))
    bbox_and_old_score = torch.tensor(pred_file_content['score'])
    pred_bboxes = bbox_and_old_score[:, :4]
    pred_cate_ids = torch.tensor(pred_file_content['category_id'])
    
    print('predict_scores', predict_scores.shape, 'pred_bboxes', pred_bboxes.shape, 'pred_cate_ids', pred_cate_ids.shape)
    # select the top 300 prediction
    top_vals, top_idxes = torch.topk(predict_scores, 300)
    pred_bboxes = pred_bboxes[top_idxes]
    pred_cate_ids = pred_cate_ids[top_idxes]
    
    for bbox, score, cate_id in zip(pred_bboxes, top_vals, pred_cate_ids):
        now_dict = {'image_id': image_id, 'bbox': bbox.tolist(), 'score': score.item(), 'category_id': cate_id.item()}
        all_final_prediction.append(now_dict)
    
# create the final prediction
final_result_path = '/home/zhuoming/final_pred_v1.json'
file = open(final_result_path, 'w')
file.write(json.dumps(all_final_prediction))
file.close()
