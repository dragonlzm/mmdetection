_base_ = './mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_base.py'

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=12, metric=['bbox', 'segm'])
