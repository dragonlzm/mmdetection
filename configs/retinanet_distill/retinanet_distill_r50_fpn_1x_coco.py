_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection_with_clip_feat.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# data pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned_base_filtered',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))


# optimizer for 8 gpu
# for 2 gpu the lr should be the 0.0025
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
model = dict(
    type='RetinaDistillNet',
    bbox_head=dict(
        type='RetinaDistillHead',
        fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_80cates.pt')),
    test_cfg=dict(
        score_thr=0.05,
        max_per_img=300))
