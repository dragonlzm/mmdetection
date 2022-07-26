_base_ = './mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48.py'

data_root = 'data/coco/'
data = dict(
    train=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_distill_base48_trained_1x_flip_reg_with_embed_train2017.pkl'),
    val=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_distill_base48_trained_1x_flip_reg_with_embed_val2017.pkl'),
    test=dict(
        proposal_file=data_root + 'proposals/mask_rcnn_distill_base48_trained_1x_flip_reg_with_embed_val2017.pkl'))