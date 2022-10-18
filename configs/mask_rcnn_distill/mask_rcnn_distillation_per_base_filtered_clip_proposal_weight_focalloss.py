_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            use_zero_bg_vector=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=0.5,
                alpha=1.0,
                loss_weight=1.0))),
    train_cfg=dict(
        rcnn=dict(
            neg_weight=0.2)),)
