_base_ = './faster_rcnn_r50_fpn_1x_coco.py'

optimizer = dict(lr=0.01)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/pretrain/resnet50-0676ba61.pth')))

