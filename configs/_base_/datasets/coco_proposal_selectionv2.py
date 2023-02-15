# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadVitProposal', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal'),
    dict(type='LoadClipPred', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal'),
    #dict(type='LoadMask', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_clip_score', 'clip_mask'],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadVitProposal', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal?'),
    dict(type='LoadClipPred', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal?'),
    #dict(type='LoadMask', file_path_prefix='/data/zhuoming/detection/coco/vitdet_proposal?'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_clip_score', 'clip_mask'],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')),
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
        pipeline=test_pipeline,
        eval_filter_empty_gt=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017_except_48base_only.json',
        img_prefix=data_root + 'train2017/',
        pipeline=test_pipeline,
        eval_filter_empty_gt=True))
evaluation = dict(interval=1, metric='proposal_selection')
