_base_ = './mask_rcnn_r50_fpn_1x_coco_class_agnostic.py'

optimizer = dict(lr=0.01)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')))

