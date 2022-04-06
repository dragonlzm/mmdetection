_base_ = './mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py'

# model settings
model = dict(
    backbone_from=dict(
        #init_cfg=dict(type='Pretrained', checkpoint="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/modified_state_dict.pth"),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/clip_vitb32_full.pth", prefix='visual.'),
        #init_cfg=dict(type='Pretrained', checkpoint="/project/nevatia_174/zhuoming/detection/pretrain/clip_vitb32_full.pth", prefix='visual.'),
        #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth", prefix='backbone.'),
        init_cfg=dict(type='Pretrained', checkpoint="/project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth", prefix='backbone.'),
        ))
