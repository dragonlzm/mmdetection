_base_ = './mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48_paramwise_grad_clip.py'

# using total batchsize 16, by using the ParamWiseGradientCumulativeOptimizerHook
optimizer = dict(lr=0.02)

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                type='DoubleRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)))

