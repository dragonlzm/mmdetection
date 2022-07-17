_base_ = [
    './rpn_r50_caffe_c4.py',
]

num_support_ways = 2
num_support_shots = 10
# model settings
model = dict(
    type='AttentionRPNTextRPN',
    backbone=dict(frozen_stages=2),
    rpn_head=dict(
        type='AttentionRPNTextHead',
        in_channels=512,
        feat_channels=1024,
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=1024,
            featmap_strides=[16]),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(
                    type='DepthWiseCorrelationAggregator',
                    in_channels=512,
                    with_fc=False)
            ]),
            clip_dim=512,
            backbone_feat_out_channels=1024,
            fg_vec_cfg=dict(fixed_param=True, load_path='data/embeddings/raw_80cates.pt'),
            num_classes=80),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=6000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))
