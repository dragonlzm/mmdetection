_base_ = './rpn_attention-rpn_r50_c4_4xb2_coco_official-base-training.py'


# for 2 gpu setting (2*2)
optimizer = dict(
    lr=0.0005,
    momentum=0.9)
#optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    warmup_iters=200, warmup_ratio=0.1, step=[
        4000,
        6000,
    ])
runner = dict(max_iters=240000)
evaluation = dict(interval=6000, metric='proposal_fast')
checkpoint_config = dict(interval=6000)

runner = dict(max_iters=6000)

# load_from = 'path of base training model'
load_from = ('data/meta_learning/rpn_attention-rpn_r50_c4_2xb2_coco_official-base-training/'
             'latest.pth')