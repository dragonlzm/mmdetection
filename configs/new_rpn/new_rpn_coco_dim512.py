_base_ = './new_rpn_coco.py'

# model settings
model = dict(
    rpn_head=dict(
        encoder=dict(
            transformerlayers=dict(
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=512,
                        num_heads=8,
                        dropout=0.1)
                ],
                ffn_cfgs=dict(
                     embed_dims=512,
                 ))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=256, normalize=True)))
