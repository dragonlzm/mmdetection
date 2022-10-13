_base_ = './mask_rcnn_distillation_with_vit_lvis_bn.py'

# learning policy
model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='SeesawLoss',
                p=0.8,
                q=2.0,
                num_classes=1203,
                loss_weight=1.0))))