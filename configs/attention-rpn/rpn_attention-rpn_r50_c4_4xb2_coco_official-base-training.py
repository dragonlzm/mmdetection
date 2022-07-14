_base_ = [
    '../_base_/datasets/query_aware/base_coco.py',
    '../_base_/schedules/few_shot_base_training_schedule.py', '../_base_/models/rpn_attention_rpn_r50_c4.py',
    '../_base_/few_shot_default_runtime.py'
]
num_support_ways = 2
num_support_shots = 10
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,    
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_train2017.json')
        ])),
    val=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_val2017.json')
    ]),
    test=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_val2017.json')
    ]),
    model_init=dict(ann_cfg=[
        dict(
            type='ann_file',
            ann_file='data/coco/annotations/instances_train2017.json')
    ]))


# for 4 gpu setting (4*2, maintaining the overall batchsize the same)
optimizer = dict(
    lr=0.004,
    momentum=0.9,
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=2.0)}))
#optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(warmup_iters=1000, warmup_ratio=0.1, step=[112000, 120000])
runner = dict(max_iters=120000)
evaluation = dict(interval=60000, metric='proposal_fast')
checkpoint_config = dict(interval=20000)
log_config = dict(interval=10)

model = dict(
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
    )
)
