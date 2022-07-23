_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_with_clip_feat.py',
    '../_base_/datasets/coco_instance_with_clip_feat.py',
]

# this config is a setting in detection which is matched with the setting we use in mmdetection
# use the 2*2 batch, with 360,000 iteration, with lr=0.005

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[240000, 320000])
runner = dict(type='IterBasedRunner', max_iters=360000)

checkpoint_config = dict(interval=20000)
evaluation = dict(interval=20000)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

