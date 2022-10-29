_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_setting.py'


# optimizer for simulating the 4*4 by accumulate the 2*2 4 times
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, 
                        type='GradientCumulativeOptimizerHook', 
                        cumulative_iters=4,
                        grad_clip=dict(max_norm=10, norm_type=2))