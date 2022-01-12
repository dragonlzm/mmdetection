_base_ = './rpn_r50_fpn_1x_coco.py'


model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='/data2/lwll/zhuoming/detection/test/rpn/latest.pth')))