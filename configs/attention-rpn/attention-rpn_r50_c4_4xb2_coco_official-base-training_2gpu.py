_base_ = 'attention-rpn_r50_c4_4xb2_coco_official-base-training.py'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# for 2 gpu setting (2*2)
optimizer = dict(lr=0.002)
#optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(warmup_iters=1000, warmup_ratio=0.1, step=[224000, 240000])
runner = dict(max_iters=240000)
evaluation = dict(interval=60000)
checkpoint_config = dict(interval=20000)
log_config = dict(interval=50)

