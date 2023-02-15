_base_ = [
    '../_base_/models/proposal_selector.py', '../_base_/datasets/coco_proposal_selection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(step=[1,2])
runner = dict(type='EpochBasedRunner', max_epochs=3)