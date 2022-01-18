_base_ = './new_rpn_coco.py'

# model settings
model = dict(
    patches_list=[2, 4, 6],
    rpn_head=dict(patches_list=[2, 4, 6]))

runner = dict(type='EpochBasedRunner', max_epochs=24)
lr_config = dict(_delete_=True, policy='step', step=[12, 22])

# for 2 gpus training
optimizer = dict(lr=0.00005)