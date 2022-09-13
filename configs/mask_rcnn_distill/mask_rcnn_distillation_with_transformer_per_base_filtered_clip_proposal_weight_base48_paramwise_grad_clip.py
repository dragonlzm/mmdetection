_base_ = './mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py'

# using total batchsize 16, by using the ParamWiseGradientCumulativeOptimizerHook
optimizer = dict(lr=0.02)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here
# optimizer_config = dict(_delete_=True, 
#                         type='ParamWiseGradientCumulativeOptimizerHook', 
#                         cumulative_iters=2,
#                         grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2) , 
#                                        other=dict(max_norm=10, norm_type=2)))

optimizer_config = dict(_delete_=True, 
                        type='ParamWiseOptimizerHook',
                        grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2) , 
                                       other=dict(max_norm=10, norm_type=2)))

# for the dataset (try the verion that do not use the weight)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned',
         num_of_rand_bbox=200, select_fixed_subset=200),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))
