_base_ = './mask_rcnn_distillation_lvis_bn_seesawloss.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(fg_vec_cfg=dict(load_path='data/embeddings/base_vision_and_text_finetuned_1203cates.pt'))))