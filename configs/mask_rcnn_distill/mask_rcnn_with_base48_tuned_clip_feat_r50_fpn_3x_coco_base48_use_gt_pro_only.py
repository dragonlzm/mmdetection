_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_use_gt_pro_only.py'

# learning policy
lr_config = dict(step=[24, 33])
runner = dict(max_epochs=36)
