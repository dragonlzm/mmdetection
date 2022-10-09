_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

model = dict(
    rpn_head=dict(
        type='TSPRPNHead',
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        loss_bbox=dict(type='L1Loss', loss_weight=2.0)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),)