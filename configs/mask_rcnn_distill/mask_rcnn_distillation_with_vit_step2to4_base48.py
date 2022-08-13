_base_ = './mask_rcnn_distillation_with_vit_base48.py'

model = dict(
    backbone=dict(
        merge_step=('merge2', 'merge3', 'merge4',)))