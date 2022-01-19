# model settings
_base_ = './new_rpn_coco_patch_gt.py'

model = dict(
    patches_list=[16],
    rpn_head=dict(patches_list=[16]))

data_root = 'data/coco/'
data = dict(
    train=dict(patches_file=data_root + 'new_assigned_gt_16_by_16_train.pt'),
    val=dict(patches_file=data_root + 'new_assigned_gt_16_by_16_val.pt'),
    test=dict(patches_file=data_root + 'new_assigned_gt_16_by_16_val.pt'))

# for 2 gpus training
optimizer = dict(lr=0.00005)