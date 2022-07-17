_base_ = './rpn_r50_caffe_c4_1x_coco.py'

classes = ('person', 'bicycle', 'car', 'motorcycle', 
'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 
'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 
'couch', 'potted plant', 'dining table', 'tv')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))