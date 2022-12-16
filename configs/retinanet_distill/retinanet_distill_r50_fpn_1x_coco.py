_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
model = dict(
    bbox_head=dict(
        type='RetinaDistillHead'),
    test_cfg=dict(
        score_thr=0.05,
        max_per_img=300))
