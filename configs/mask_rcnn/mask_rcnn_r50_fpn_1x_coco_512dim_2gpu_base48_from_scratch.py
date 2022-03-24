_base_ = './mask_rcnn_r50_fpn_1x_coco_512dim_2gpu_base48.py'

# model settings
model = dict(
    backbone=dict(
        init_cfg=None))

# stage 1 setting
lr_config = dict(step=[])
runner = dict(type='EpochBasedRunner', max_epochs=32)