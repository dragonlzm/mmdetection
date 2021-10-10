_base_ = [
    '../_base_/models/classification_transfer_model_c4_no_avg_pooling.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]'''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

NOVEL_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'bottle', 'chair', 'couch', 'potted plant',
                'dining table', 'tv')

dataset_type = 'CocoFewshotTestDataset'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(type=dataset_type,
                pipeline=train_pipeline,
                classes=NOVEL_CLASSES),
    val=dict(type=dataset_type,
                pipeline=test_pipeline,
                classes=NOVEL_CLASSES),
    test=dict(type=dataset_type,
                pipeline=test_pipeline,
                classes=NOVEL_CLASSES))

checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=25,
    warmup_ratio=0.001,
    step=[3, 5])
runner = dict(type='EpochBasedRunner', max_epochs=6)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])