_base_ = './mask_rcnn_distillation_with_vit_lvis_base_range_scale.py'

# learning policy
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=866,
                loss_weight=1.0))))

# learning policy
lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=24, metric=['bbox', 'segm'])