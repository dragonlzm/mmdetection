_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_setting.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_shared_convs=2)))
