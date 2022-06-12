_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro.py'

# learning policy
lr_config = dict(step=[32, 44])
runner = dict(max_epochs=48)
