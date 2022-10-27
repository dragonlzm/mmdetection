_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_setting.py'

# learning policy
lr_config = dict(policy='step', step=[9,11])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
