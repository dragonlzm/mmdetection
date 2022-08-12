_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_detectron_2x2_360k.py'
# this config is a setting in detection which is matched with the setting we use in mmdetection
# use the 2*4 batch, with 180,000 iteration, with lr=0.01


optimizer = dict(type='SGD', lr=0.01)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[120000, 160000])
runner = dict(type='IterBasedRunner', max_iters=180000)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

checkpoint_config = dict(interval=10000)
evaluation = dict(interval=10000)