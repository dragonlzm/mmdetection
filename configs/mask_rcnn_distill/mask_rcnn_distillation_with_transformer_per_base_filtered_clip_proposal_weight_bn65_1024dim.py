_base_ = './mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_bn65.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            fc_out_channels=1024,
            encoder=dict(
                transformerlayers=dict(
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