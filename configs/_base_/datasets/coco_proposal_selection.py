# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    #dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadCLIPProposal', file_path_prefix='data/coco/clip_proposal/32_32_512'),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_scores']),
]

test_pipeline = [
    #dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadCLIPProposal', file_path_prefix='data/coco/clip_proposal/32_32_512'),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_scores']),
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_48base_only.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_except_48base_only.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_except_48base_only.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='proposal_selection')
