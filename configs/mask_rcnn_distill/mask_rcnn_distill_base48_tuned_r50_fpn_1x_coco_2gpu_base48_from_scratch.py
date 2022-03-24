_base_ = './mask_rcnn_distill_base48_tuned_r50_fpn_1x_coco_2gpu_base48.py'

# model settings
model = dict(
    backbone_to=dict(
        init_cfg=None))

# stage 1 setting
lr_config = dict(step=[])
runner = dict(type='EpochBasedRunner', max_epochs=32)