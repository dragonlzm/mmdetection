_base_ = './mask_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(lr=0.01)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')),
    neck=dict(out_channels=512),
    rpn_head=dict(in_channels=512),
    roi_head=dict(
        bbox_roi_extractor=dict(out_channels=512),
        bbox_head=dict(in_channels=512),
        mask_roi_extractor=dict(out_channels=512),
        mask_head=dict(in_channels=512)),)