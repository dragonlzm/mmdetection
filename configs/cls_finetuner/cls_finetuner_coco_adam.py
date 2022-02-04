_base_ = './cls_finetuner_coco.py'

model = dict(backbone=dict(open_ln=True))

optimizer = dict(
    _delete_=True,    
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(_delete_=True, policy='step', step=[])

runner = dict(max_epochs=12)