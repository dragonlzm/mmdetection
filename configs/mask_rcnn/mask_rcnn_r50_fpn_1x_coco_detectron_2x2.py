_base_ = './mask_rcnn_r50_fpn_1x_coco_detectron_2x8.py'

optimizer = dict(type='SGD', lr=0.005)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[240000, 320000])
runner = dict(type='IterBasedRunner', max_iters=360000)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)