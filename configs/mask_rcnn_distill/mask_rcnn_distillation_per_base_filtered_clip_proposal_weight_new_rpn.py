_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

model = dict(
    rpn_head=dict(
        type='TSPRPNHead',
        in_channels=256,
        feat_channels=256,
        num_convs=2,
        loss_bbox=dict(type='L1Loss', loss_weight=2.0)))