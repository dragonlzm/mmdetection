_base_ = './mask_rcnn_distill_r50_fpn_1x_coco.py'

# model settings
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint="data/test/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_5/epoch_12.pth"),
    roi_head=dict(
        type='StandardRoIHeadCLIPCls',
        extra_backbone=dict(
            type='myVisionTransformer',
            input_resolution=224,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            output_dim=512,
            fixed_param=True,
            #init_cfg=dict(type='Pretrained', checkpoint="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")),
            #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/modified_state_dict.pth"),
            #init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/data/pretrained/clip_vitb32_full.pth", prefix='visual.'),
            init_cfg=dict(type='Pretrained', checkpoint="data/test/cls_finetuner_clip_base_all_train/epoch_12.pth", prefix='backbone.'),
        )))