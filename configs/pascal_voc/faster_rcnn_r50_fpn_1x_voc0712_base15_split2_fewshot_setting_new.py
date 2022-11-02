_base_ = './faster_rcnn_r50_fpn_1x_voc0712_base15_split1_fewshot_setting_new.py'

classes = ('bicycle', 'bird', 'boat', 'bus', 'car', 'cat',
            'chair', 'diningtable', 'dog', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor')
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))