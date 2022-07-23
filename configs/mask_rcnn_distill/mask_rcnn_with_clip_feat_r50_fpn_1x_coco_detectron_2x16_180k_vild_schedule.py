_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_detectron_2x16_180k.py'
# the setting in this config is not matched with the previous experiment in mmdetection
# with longer training schedule, it should be 45000 iters if want to match with the previous exp
# we following the iteration number in vild paper and make the batch size larger to see what will happen

# learning policy
lr_config = dict(step=[162000, 171000, 175500])