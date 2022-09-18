_base_ = './mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48_paramwise_grad_clip_exp.py'

# using total batchsize 16, by using the ParamWiseGradientCumulativeOptimizerHook
optimizer = dict(lr=0.005)

optimizer_config = dict(_delete_=True, 
                        type='ParamWiseGradientCumulativeOptimizerHook', 
                        cumulative_iters=2,
                        grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2)))
