_base_ = './retinanet_distill_r50_fpn_1x_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=48,
        fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_48cates.pt')))
