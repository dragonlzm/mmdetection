_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here
optimizer = dict(lr=0.005)

# optimizer = dict(type='HybridOptimizer', lr=0.005, momentum=0.9, weight_decay=0.0001,
#                  constructor='HybridOptimizerConstructor', tranformer_multiplier=0.005)

# optimizer_config = dict(_delete_=True, 
#                         #type='ParamWiseOptimizerHook', 
#                         grad_clip=dict(max_norm=10, norm_type=2))

optimizer_config = dict(_delete_=True, 
                        type='ParamWiseGradientCumulativeOptimizerHook', 
                        cumulative_iters=1,
                        grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2), 
                                       other=dict(max_norm=10, norm_type=2)))


# model settings
model = dict(
    roi_head=dict(
        type='StandardRoIHeadDistillWithTransformer',
        bbox_head=dict(
            type='TransformerEmbeddingBBoxHead',
            num_shared_convs=0,
            num_shared_fcs=1,
            with_avg_pool=False,
            reg_class_agnostic=True,
            reg_with_cls_embedding=False,
            fc_out_channels=512,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=512,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=512,
                            feedforward_channels=2048,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))))),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='DoubleRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)))