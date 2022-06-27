_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_bn65.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_65cates_new.pt'))))
