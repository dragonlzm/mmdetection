_base_ = './rpn_r50_fpn_1x_lvis_base.py'


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=12)