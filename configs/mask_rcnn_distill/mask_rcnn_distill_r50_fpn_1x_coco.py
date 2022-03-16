_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_distillation.py',
    '../_base_/datasets/coco_instance_with_distillation.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

#optimizer = dict(lr=0.01)

#model = dict(
#    backbone=dict(
#        init_cfg=dict(type='Pretrained', 
#            checkpoint='data/pretrain/resnet50-0676ba61.pth')))