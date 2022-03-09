_base_ = './cls_finetuner_clip_full_coco.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='GenerateCroppedPatches', use_rand_bboxes=False),
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'cropped_patches']),
]


data = dict(workers_per_gpu=4,
    val=dict(eval_filter_empty_gt=True, pipeline=test_pipeline),
    test=dict(eval_filter_empty_gt=True, pipeline=test_pipeline))