_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_shared_convs=0,
            num_shared_fcs=1,
            with_avg_pool=False,
            reg_class_agnostic=True)))