_base_ = './cls_finetuner_coco.py'

model = dict(backbone=dict(open_ln=True))

paramwise_cfg = dict(custom_keys={
                '.backbone': dict(lr_mult=0.001)})