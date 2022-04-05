_base_ = [
    '../_base_/models/mask_rcnn_with_clip_feat_r50_fpn_1x_coco.py',
    '../_base_/datasets/coco_instance_with_clip_feat.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]