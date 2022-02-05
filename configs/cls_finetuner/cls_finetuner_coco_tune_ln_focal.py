_base_ = './cls_finetuner_coco_tune_ln.py'

# model settings
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)
        )
        )


