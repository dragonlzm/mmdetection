_base_ = './faster_rcnn_r50_fpn_1x_voc0712_base15_split1_fewshot_setting.py'

# for 2*2 batch size
lr_config = dict(warmup_iters=100, step=[48000, 64000])
runner = dict(max_iters=72000)
evaluation = dict(interval=12000, metric='mAP')
checkpoint_config = dict(interval=12000)
