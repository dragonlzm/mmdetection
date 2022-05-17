_base_ = 'mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py'

# model settings
model = dict(
    rpn_head=dict(
        type='TriWayRPNHead',
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(
        rpn_proposal=dict(
            score_thr=0.0)))
