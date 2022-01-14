_base_ = './new_rpn_coco.py'

# model settings
model = dict(
    rpn_head=dict(
        encoder=dict(
            transformerlayers=dict(
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                 ffn_cfgs=dict(
                     embed_dims=256,
                     feedforward_channels=2048)
                    ))))
