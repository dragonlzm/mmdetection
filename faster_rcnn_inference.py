from mmdet.apis import init_detector, inference_detector
import mmcv
from os import listdir
from os.path import isfile, join
import torch
import json

# obtain all validation image under the directory
#mypath = '/data2/lwll/zhuoming/detection/coco/val2017'
#onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# prepare the file list
json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
# load the json file
json_val = json.load(open(json_file_path))

# aggregate the annotation for each image
file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
from_img_id_to_bbox = {}
#{image_id:{image_name:"", bbox_list:[]},}
# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    box = anno['bbox']
    #print('before convert', box)
    #box.append(anno['category_id'])
    new_bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    #print('new box', new_bbox)
    box = torch.tensor(new_bbox).unsqueeze(dim=0)
    #print('after convert', box)
    from_img_id_to_bbox[image_id]['bbox'].append(box)
    #break

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
#img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
final_feat_list = []
bbox_feat_list = []

for i, image_id in enumerate(from_img_id_to_bbox.keys()):
    file_path = from_img_id_to_bbox[image_id]['path']
    #if len(from_img_id_to_bbox[image_id]['bbox']) == 0:
    #    continue
    #result = inference_detector(model, file_path, gt_bbox=[[torch.cat(from_img_id_to_bbox[image_id]['bbox'], dim=0)]])
    result = inference_detector(model, file_path)
    final_feat_list.append(result['final_feat'])
    #bbox_feat_list.append(result['bbox_feats'].reshape([result['bbox_feats'].shape[0],-1]))
    #break
    if i % 100 == 0:
        print(i)

# visualize the results in a new window
#model.show_result(img, result)
# or save the visualization results to image files
#model.show_result(img, result, out_file='result.jpg')
final_feat = torch.cat(final_feat_list, dim=0)
#bbox_feat = torch.cat(bbox_feat_list, dim=0)

print(final_feat.shape)
#print(bbox_feat.shape)

#torch.save(final_feat.cpu(), 'fasterrcnn_final_feat_gt.pt')

torch.save(final_feat.cpu(), 'fasterrcnn_4_by_4_final_feat.pt')
#torch.save(bbox_feat.cpu(), 'fasterrcnn_4_by_4_bbox_feat.pt')