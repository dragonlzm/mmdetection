_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_with_clip_feat.py',
    '../_base_/datasets/coco_instance_with_clip_feat.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# set for 2 gpu training (the lr is match with vild paper setting)
optimizer = dict(lr=0.005)
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')))