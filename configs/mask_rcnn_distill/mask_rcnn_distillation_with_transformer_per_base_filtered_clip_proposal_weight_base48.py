_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here

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
