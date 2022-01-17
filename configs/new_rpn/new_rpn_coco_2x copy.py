_base_ = './new_rpn_coco.py'

runner = dict(type='EpochBasedRunner', max_epochs=24)
lr_config = dict(_delete_=True, policy='step', step=[16, 22])