_base_ = './fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48.py'
# using random proposal with clip feat
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/coco/clip_proposal_feat/base48_finetuned_base_filtered',
         num_of_rand_bbox=200, select_fixed_subset=200),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline))

# optimizer
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', 
#             checkpoint='open-mmlab://detectron/resnet50_caffe')))