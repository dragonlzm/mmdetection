_base_ = 'deformable_detr_r50_16x2_50e_coco.py'

# this config is for training the model using 2x2 setting
# but need to simulate the 16*2 setting
optimizer_config = dict(type='GradientCumulativeOptimizerHook', 
                        cumulative_iters=8)