_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro_reg_with_embedding.py'

# with filp become a default setting
# model settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/rpn_proposal_feat/raw',
         num_of_rand_bbox=200, select_fixed_subset=200, load_rand_bbox_weight=True),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats', 'rand_bbox_weights']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            fg_vec_cfg=dict(fixed_param=True, load_path='data/embeddings/raw_48cates.pt'))))