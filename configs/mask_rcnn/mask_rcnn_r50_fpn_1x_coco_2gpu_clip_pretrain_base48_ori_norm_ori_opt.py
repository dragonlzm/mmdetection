_base_ = './mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm.py'

# optimizer
optimizer = dict(_delete_=True,
                 type='SGD', 
                 lr=0.01, 
                 momentum=0.9, 
                 weight_decay=0.0001)
