_base_ = './faster_rcnn_r50_fpn_1x_voc0712_base15_split1_fewshot_setting_new.py'

classes = ('aeroplane', 'bicycle', 'bird', 'bottle', 'bus',
            'car', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'person', 'pottedplant', 'train', 'tvmonitor')
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))