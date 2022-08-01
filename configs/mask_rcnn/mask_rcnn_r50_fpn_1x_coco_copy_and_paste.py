_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu.py'

# dataset settings

# if self.poly2mask = False in LoadAnnotations, please set self.poly2mask = False in CopyPaste
# if channel_order is set in LoadImageFromFile, please set it in the CopyPaste

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CopyPaste'),    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data = dict(train=dict(pipeline=train_pipeline, copy_and_paste=True))
evaluation = dict(metric=['bbox', 'segm'])
