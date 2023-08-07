_base_ = './mask_rcnn_distillation_lvis_raw_fc866.py'

# learning policy
lr_config = dict(step=[74, 79])
runner = dict(type='EpochBasedRunner', max_epochs=84)
evaluation = dict(interval=96, metric=['bbox', 'segm'])