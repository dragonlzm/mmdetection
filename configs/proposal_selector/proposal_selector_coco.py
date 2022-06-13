_base_ = [
    '../_base_/models/proposal_selector.py', '../_base_/datasets/coco_proposal_selection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

lr_config = dict(step=[])
runner = dict(type='EpochBasedRunner', max_epochs=6)