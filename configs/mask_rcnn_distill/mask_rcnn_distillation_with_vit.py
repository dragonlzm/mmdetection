_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro_reg_with_embedding.py'

# model settings
model = dict(
    backbone=dict(
        type='ResNetWithVit',
        clip_architecture="ViT-B/32"))

