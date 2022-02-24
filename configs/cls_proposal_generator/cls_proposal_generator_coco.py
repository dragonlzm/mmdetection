_base_ = [
    '../_base_/models/cls_proposal_generator.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            #dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], 
                    meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction')),
        ])
]'''

classes = ('person', 'bicycle', 'car', 'motorcycle', 
'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
'couch', 'potted plant', 'dining table', 'tv')

data_root = 'data/coco/'
data = dict(train=dict(pipeline=train_pipeline),
            val=dict(eval_filter_empty_gt=True, 
                pipeline=test_pipeline, 
                ann_file=data_root + 'annotations/train_1shots.json',
                img_prefix=data_root + 'train2017/',
                classes=classes),
            test=dict(eval_filter_empty_gt=True, 
                pipeline=test_pipeline, 
                ann_file=data_root + 'annotations/train_1shots.json',
                img_prefix=data_root + 'train2017/',
                classes=classes))
evaluation = dict(interval=1, metric='clip_proposal')

lr_config = dict(step=[])
runner = dict(type='EpochBasedRunner', max_epochs=6)