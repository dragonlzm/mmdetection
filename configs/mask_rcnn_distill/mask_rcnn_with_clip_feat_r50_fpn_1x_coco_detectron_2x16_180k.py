_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_detectron_2x2_360k.py'
# the setting in this config is not matched with the previous experiment in mmdetection
# with longer training schedule, it should be 45000 iters if want to match with the previous exp
# we following the iteration number in vild paper and make the batch size larger to see what will happen

optimizer = dict(type='SGD', lr=0.04)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[120000, 160000])
runner = dict(type='IterBasedRunner', max_iters=180000)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16)

checkpoint_config = dict(interval=5000)
evaluation = dict(interval=5000)