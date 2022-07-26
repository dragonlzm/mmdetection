_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple.py',
    '../_base_/datasets/coco_instance_with_clip_feat.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# set for 2 gpu training (the lr is match with vild paper setting)
optimizer = dict(lr=0.005)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_with_cls_embedding=True,
            fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_80cates.pt'))))


# directly use the setting, with finetuned feature, with flipping, with text embedding for augmentation
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
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
            dict(type='ToTensor', keys=['proposals']),
            dict(
                type='ToDataContainer',
                fields=[dict(key='proposals', stack=False)]),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_base48_trained_1x_cls_agno_train2017.pkl',
        pipeline=train_pipeline),
    val=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_base48_trained_1x_cls_agno_val2017.pkl',
        pipeline=test_pipeline),
    test=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_base48_trained_1x_cls_agno_val2017.pkl',
        pipeline=test_pipeline))
