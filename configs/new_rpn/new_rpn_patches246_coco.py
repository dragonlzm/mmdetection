_base_ = './new_rpn_coco.py'

# model settings
model = dict(
    patches_list=[2, 4, 6],
    rpn_head=dict(patches_list=[2, 4, 6]))


runner = dict(type='EpochBasedRunner', max_epochs=24)
lr_config = dict(_delete_=True, policy='step', step=[16, 22])