_base_ = './mask_rcnn_distillation_voc_split1_base.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_with_cls_embedding=True)))