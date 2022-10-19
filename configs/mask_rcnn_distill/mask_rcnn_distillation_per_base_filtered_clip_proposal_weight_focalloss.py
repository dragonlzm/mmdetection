_base_ = './mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            use_zero_bg_vector=True,
            loss_cls=dict(
                type='my_focal_loss'))))
