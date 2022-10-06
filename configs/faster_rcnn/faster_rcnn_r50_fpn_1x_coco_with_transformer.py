_base_ = './faster_rcnn_r50_fpn_1x_coco.py'

# set for simulating bs 16
#optimizer = dict(lr=0.02)
optimizer = dict(_delete_=True,
    type='HybridOptimizer', lr=0.02, momentum=0.9, weight_decay=0.0001,
    constructor='HybridOptimizerConstructor', tranformer_multiplier=0.005)

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here, defualt using 2gpu
optimizer_config = dict(_delete_=True, 
                        type='ParamWiseGradientCumulativeOptimizerHook', 
                        cumulative_iters=4,
                        grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2)))

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', 
            checkpoint='data/pretrain/resnet50-0676ba61.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    rpn_head=dict(
        type='TSPRPNHead',
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        loss_bbox=dict(type='L1Loss', loss_weight=2.0)),
    roi_head=dict(
        type='StandardRoIHeadWithTransformer',
        bbox_head=dict(
            type='TransformerBBoxHead',
            num_shared_convs=0,
            num_shared_fcs=1,
            with_avg_pool=False,
            reg_class_agnostic=False,
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
            debug=False),
        rpn_proposal=dict(
            max_per_img=700)),
    test_cfg=dict(
        rpn=dict(
            max_per_img=700)))
