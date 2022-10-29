_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_setting.py'

# optimizer for 4*4
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)