_base_ = './mask_rcnn_distillation_voc_split1_base_r101_fewshot_setting.py'

classes = ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'chair', 'diningtable', 'dog', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor')

data = dict(
    train=dict(dataset=dict(classes=classes)),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(fg_vec_cfg=dict(fixed_param=True,
                        load_path='data/embeddings/base_finetuned_voc_split2_15cates.pt'))))
