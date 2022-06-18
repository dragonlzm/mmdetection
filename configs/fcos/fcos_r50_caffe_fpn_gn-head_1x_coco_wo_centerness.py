_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'
# model settings
model = dict(
    bbox_head=dict(type='MyFCOSHead', with_centerness=False))