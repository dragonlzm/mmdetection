_base_ = './mask_rcnn_distillation_with_vit_base48.py'

# modify the train config
# model settings
model = dict(
    train_cfg=dict(
        rcnn=dict(use_only_gt_pro_for_distill=True)))
