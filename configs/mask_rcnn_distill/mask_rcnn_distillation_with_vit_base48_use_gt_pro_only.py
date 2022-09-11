_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py'

# modify the train config
# model settings
model = dict(
    train_cfg=dict(
        rcnn=dict(use_only_gt_pro_for_distill=True)))
