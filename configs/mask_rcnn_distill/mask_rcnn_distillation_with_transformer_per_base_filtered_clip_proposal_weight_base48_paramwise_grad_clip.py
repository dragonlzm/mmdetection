_base_ = './mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py'

# using total batchsize 16, by using the ParamWiseGradientCumulativeOptimizerHook
optimizer = dict(lr=0.02)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# regression with embedding, base filtered proposal, per distillation bbox weight
# become default setting in here
optimizer_config = dict(_delete_=True, 
                        type='ParamWiseGradientCumulativeOptimizerHook', 
                        cumulative_iters=2,
                        grad_clip=dict(encoder=dict(max_norm=0.01, norm_type=2) , 
                                       other=dict(max_norm=10, norm_type=2)))