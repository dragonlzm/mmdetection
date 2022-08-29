_base_ = './cls_finetuner_clip_lvis_base_train.py'

model = dict(
    rpn_head=dict(
        open_ln=True,
        use_gt_name=True))
