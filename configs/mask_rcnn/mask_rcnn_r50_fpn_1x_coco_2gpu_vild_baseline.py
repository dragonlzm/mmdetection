_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4)
        ))
