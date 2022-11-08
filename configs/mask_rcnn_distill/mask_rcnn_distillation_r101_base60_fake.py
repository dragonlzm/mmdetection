_base_ = './mask_rcnn_distillation_base60_fake.py'

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='/data/zhuoming/detection/pretrain/resnet101-63fe2227.pth')))