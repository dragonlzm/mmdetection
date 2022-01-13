_base_ = './new_rpn_coco.py'

# model settings
model = dict(
    patches_list=[2, 4, 6],
    rpn_head=dict(patches_list=[2, 4, 6]))
