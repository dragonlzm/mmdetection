_base_ = './proposal_selector_coco.py'

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadCLIPProposal', file_path_prefix='data/coco/base48_only_train_prediction'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_scores'],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='LoadCLIPProposal', file_path_prefix='data/coco/bn65_val_prediction'),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Normalize', **img_norm_cfg),
    #dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'proposal_bboxes', 'proposal_scores'],
            meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape')),
]

classes_48 = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

classes_65 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')


data = dict(
    train=dict(
        classes=classes_48,
        pipeline=train_pipeline),
    val=dict(
        classes=classes_65,
        pipeline=test_pipeline,
        eval_filter_empty_gt=True),
    test=dict(
        classes=classes_65,
        pipeline=test_pipeline,
        eval_filter_empty_gt=True))
evaluation = dict(interval=1, metric='proposal_selection')
