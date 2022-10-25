_base_ = './faster_rcnn_r50_fpn_1x_voc0712.py'

# default bs = 2x2
# dataset settings
classes = ('aeroplane', 'bicycle', 'boat', 'bottle', 'car',
            'cat', 'chair', 'diningtable', 'dog', 'horse',
            'person', 'pottedplant', 'sheep', 'train',
            'tvmonitor')

### repeat daatset
data = dict(
    train=dict(dataset=dict(classes=classes)),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=15)))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_shared_convs=4,
            reg_class_agnostic=True,
            num_classes=15)))

# training schdule
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12

