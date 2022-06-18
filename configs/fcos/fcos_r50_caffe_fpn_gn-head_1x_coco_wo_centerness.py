_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'
# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50_caffe-788b5fa3.pth')),
    bbox_head=dict(type='MyFCOSHead', with_centerness=False))