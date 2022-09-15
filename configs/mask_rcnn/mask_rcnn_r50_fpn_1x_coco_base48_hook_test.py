_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py'

# here we use the cumulated hook to check whether the hook is working
# using 4 gpu for training
# simulating the batch size 16
optimizer = dict(lr=0.02)


optimizer_config = dict(_delete_=True, 
                        type='ParamWiseGradientCumulativeOptimizerHook', 
                        cumulative_iters=2,
                        grad_clip=None)
