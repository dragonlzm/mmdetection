_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_albu_augmentation.py'

# learning policy
lr_config = dict(step=[24, 33])
runner = dict(max_epochs=36)