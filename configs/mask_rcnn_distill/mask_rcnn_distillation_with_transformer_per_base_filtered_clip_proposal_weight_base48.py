_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here
optimizer = dict(type='HybridOptimizer', lr=0.005, momentum=0.9, weight_decay=0.0001,
                 constructor='HybridOptimizerConstructor', tranformer_multiplier=0.01)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

optimizer_config = dict(_delete_=True, 
                        type='ParamWiseOptimizerHook', 
                        grad_clip=dict(max_norm=10, norm_type=2))

# model settings
model = dict(
    roi_head=dict(
        type='StandardRoIHeadDistillWithTransformer',
        bbox_head=dict(
            type='TransformerBBoxHead',
            num_shared_convs=2,
            reg_class_agnostic=True,
            in_channels=256,
            fc_out_channels=1024,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=1024,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=1024,
                            feedforward_channels=2048,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))))))
