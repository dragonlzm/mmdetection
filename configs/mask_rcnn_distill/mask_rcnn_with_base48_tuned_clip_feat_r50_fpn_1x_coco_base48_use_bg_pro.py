_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    #dict(type='LoadCLIPFeat', file_path_prefix='data/coco/feat/base48_finetuned',
    #     num_of_rand_bbox=100),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200),
    dict(type='LoadCLIPBGProposal', file_path_prefix='data/coco/clip_bg_proposal_feat/base48_finetuned/random',
         num_of_rand_bbox=20, select_fixed_subset=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'bg_bboxes', 'bg_feats']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

# modify the train config
# model settings
model = dict(
    train_cfg=dict(
        rcnn=dict(use_bg_pro_as_ns=True)))