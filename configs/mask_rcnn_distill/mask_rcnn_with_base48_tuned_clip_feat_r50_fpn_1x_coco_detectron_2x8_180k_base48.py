# this config is the model that has the best performance in all exp but with detectron setting
_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_detectron_2x8_180k.py'

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
    train=dict(classes=classes, pipeline=train_pipeline),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=48,
                       reg_with_cls_embedding=True,
                       fg_vec_cfg=dict(fixed_param=True, 
                                       #load_path='/data2/lwll/zhuoming/detection/embeddings/base_finetuned_48cates.pt',
                                       load_path='data/embeddings/base_finetuned_48cates.pt')),
        mask_head=dict(num_classes=48)))