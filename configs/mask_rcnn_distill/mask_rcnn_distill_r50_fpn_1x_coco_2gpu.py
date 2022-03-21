_base_ = './mask_rcnn_distill_r50_fpn_1x_coco.py'

optimizer = dict(lr=0.01)

model = dict(
    backbone_to=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')))