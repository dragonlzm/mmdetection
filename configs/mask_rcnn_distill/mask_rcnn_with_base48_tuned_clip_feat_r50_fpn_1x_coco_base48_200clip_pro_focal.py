_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro.py'


# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            use_bg_vector=False,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0))))

        