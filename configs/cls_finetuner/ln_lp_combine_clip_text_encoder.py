_base_ = './cls_finetuner_clip_full_coco.py'


model = dict(
    type='ClsFinetuner',
    neck=None,
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', 
        checkpoint="/data2/lwll/zhuoming/detection/test/cls_finetuner_coco_base_all_tune_ln_focal/epoch_12.pth",
        prefix='backbone.')),
    rpn_head=dict(
        init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/clip_vitb32_full.pth"),
        #init_cfg=dict(type='Pretrained', checkpoint="/project/nevatia_174/zhuoming/detection/pretrain/clip_vitb32_full.pth"),        
        ))