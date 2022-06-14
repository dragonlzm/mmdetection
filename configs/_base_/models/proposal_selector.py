model = dict(
    type='ProposalSelector',
    encoder=dict(
        type='DetrTransformerEncoder',
        num_layers=6,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.1)
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
    loss=dict(type='L1Loss'),
    input_dim=5)