# model settings
model = dict(
    type='NEWRPN',
    patches_list=[4],
    #patches_list=[2, 4, 6],
    neck=None,
    backbone=dict(
        type='myVisionTransformer',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        #init_cfg=dict(type='Pretrained', checkpoint="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/modified_state_dict.pth"),
        init_cfg=dict(type='Pretrained', checkpoint="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/modified_state_dict.pth"),
        fixed_param=True),

    rpn_head=dict(
        type='EncoderHead',
        num_classes=1,
        in_channels=512,
        patches_list=[4],
        #patches_list=[2, 4, 6],
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=512,
                     num_fcs=2,
                     ffn_drop=0.1,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
            #loss_cls=dict(
            #    type='CrossEntropyLoss',
            #    bg_cls_weight=0.1,
            #    use_sigmoid=False,
            #    loss_weight=1.0,
            #    class_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        rpn_head=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))
                )
            )
        )
