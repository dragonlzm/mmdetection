_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_bn65.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            use_bg_vector=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))))
