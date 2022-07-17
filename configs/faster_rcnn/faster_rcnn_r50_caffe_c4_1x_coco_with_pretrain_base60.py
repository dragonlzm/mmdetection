_base_ = './faster_rcnn_r50_caffe_c4_1x_coco_with_pretrain.py'

classes = ('truck', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
            'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

# make model head = 60
# model settings
model = dict(
    backbone=dict(
            init_cfg=dict(type='Pretrained', 
                checkpoint='data/pretrain/resnet50-0676ba61.pth')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=60)))

