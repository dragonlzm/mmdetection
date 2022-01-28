# model settings
model = dict(
    type='ClsFinetuner',
    patches_list=[4],
    #patches_list=[2, 4, 6],
    neck=None,
    backbone=dict(
        type='myVisionTransformer',
        input_resolution=224,
        patch_size=32,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
        #init_cfg=dict(type='Pretrained', checkpoint="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")),
        init_cfg=dict(type='Pretrained', checkpoint="/data2/lwll/zhuoming/code/new_rpn/mmdetection/modified_state_dict.pth"),
        #init_cfg=dict(type='Pretrained', checkpoint="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/modified_state_dict.pth"),
        fixed_param=True),

    rpn_head=dict(
        type='ClipClsHead',
        num_classes=80,
        in_channels=512,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        word_embeddings_path = None,
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            loss_weight=1.0)),
        )
