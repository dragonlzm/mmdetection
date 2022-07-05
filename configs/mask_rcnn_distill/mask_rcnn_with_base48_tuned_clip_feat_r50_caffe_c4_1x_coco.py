_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_with_clip_feat.py',
    '../_base_/datasets/coco_instance_with_clip_feat.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# set for 2 gpu training
optimizer = dict(lr=0.005)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='LoadCLIPFeat', file_path_prefix='/data/zhuoming/detection/coco/feat/raw',
    #     num_of_rand_bbox=20),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/feat/base48_finetuned',
         num_of_rand_bbox=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

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
]

data = dict(
    train=dict(
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_80cates.pt'))))


