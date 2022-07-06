_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco.py'

img_norm_cfg = dict(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], to_rgb=True, divide_255=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='LoadCLIPFeat', file_path_prefix='data/coco/feat/base48_finetuned',
    #     num_of_rand_bbox=100),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/raw_rn50_500',
         num_of_rand_bbox=200, select_fixed_subset=200),
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
    train=dict(pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline),
    test=dict(
        pipeline=test_pipeline))

model = dict(
    backbone=dict(
        deep_stem=True,
        init_cfg=dict(type='Pretrained', checkpoint="data/pretrain/clip_rn50_full_name_modified.pth", prefix='visual.')),
    roi_head=dict(
        bbox_head=dict(
            fg_vec_cfg=dict(fixed_param=True,
                            load_path='data/embeddings/raw_rn50_80cates.pt'))))

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

# optimizer
optimizer = dict(type='SGD', 
                 lr=0.005, 
                 momentum=0.9, 
                 weight_decay=0.0001,
                 paramwise_cfg=dict(
                 custom_keys={'backbone': dict(lr_mult=0.02, decay_mult=1.0)}))
