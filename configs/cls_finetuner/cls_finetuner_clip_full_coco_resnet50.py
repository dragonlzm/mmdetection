_base_ = './cls_finetuner_clip_full_coco.py'

# change the classification loss
# add the text encoder module
# fixed the all parametere except the ln of the text encoder
model = dict(
    backbone=dict(_delete_=True, 
        type='ModifiedResNet',
        layers=(3, 4, 6, 3), 
        output_dim=1024, 
        heads=32, 
        input_resolution=224, 
        width=64,
        init_cfg=dict(type='Pretrained', checkpoint="data/pretrain/clip_rn50_full.pth", prefix='visual.')),
    rpn_head=dict(
        init_cfg=dict(type='Pretrained', checkpoint="data/pretrain/clip_rn50_full.pth")))