_base_ = './mask_rcnn_r50_fpn_1x_coco.py'


# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(reg_class_agnostic=True),
        mask_head=dict(class_agnostic=True)))
