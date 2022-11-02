_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_dataset_rpn_proposal_raw.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='LoadCLIPFeat', file_path_prefix='data/VOCdevkit/rpn_proposal_feat/split1_raw',
         num_of_rand_bbox=200, select_fixed_subset=200),    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_feats',
                               'rand_bboxes', 'rand_feats']),
]

data = dict(train=dict(pipeline=train_pipeline))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_with_cls_embedding=False,
            fg_vec_cfg=dict(fixed_param=True,
                        load_path='data/embeddings/raw_voc_split1_15cates.pt'))))