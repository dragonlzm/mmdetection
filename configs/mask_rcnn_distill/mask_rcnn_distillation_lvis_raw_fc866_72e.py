_base_ = './mask_rcnn_distillation_lvis_raw_fc866.py'

# learning policy
lr_config = dict(step=[48, 66])
runner = dict(type='EpochBasedRunner', max_epochs=72)
evaluation = dict(interval=80, metric=['bbox', 'segm'])