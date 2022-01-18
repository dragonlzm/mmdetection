# model settings
_base_ = './new_rpn_coco_patch_gt.py'

model = dict(
    patches_list=[8],
    rpn_head=dict(patches_list=[8]))

data_root = 'data/coco/'
data = dict(
    train=dict(patches_file=data_root + 'new_assigned_gt_8_by_8_train.pt'),
    val=dict(patches_file=data_root + 'new_assigned_gt_8_by_8_val.pt'),
    test=dict(patches_file=data_root + 'new_assigned_gt_8_by_8_val.pt'))

# for 2 gpus training
optimizer = dict(lr=0.00005)