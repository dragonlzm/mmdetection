_base_ = './new_rpn_coco.py'

data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'patches_gt']),
]

data = dict(
    train=dict(patches_file=data_root + 'new_assigned_gt_4_by_4_train.pt',
                pipeline=train_pipeline),
    val=dict(patches_file=data_root + 'new_assigned_gt_4_by_4_val.pt'),
    test=dict(patches_file=data_root + 'new_assigned_gt_4_by_4_val.pt'))

# change the evalution
evaluation = dict(interval=1, metric='acc')

# model settings
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0),
        loss_bbox=None,
        loss_iou=None))

# for 2 gpus training
optimizer = dict(lr=0.00005)
lr_config = dict(_delete_=True, policy='step', step=[4, 8])