_base_ = './rpn_r50_fpn_1x_coco.py'

train_class = ("toilet","bicycle","apple","train","laptop",
"carrot","motorcycle","oven","chair","mouse","boat","kite",
"sheep","horse","sandwich","clock","tv","backpack","toaster",
"bowl","microwave","bench","book","orange","bird","pizza","fork",
"frisbee","bear","vase","toothbrush","spoon","giraffe","handbag",
"broccoli","refrigerator","remote","surfboard","car","bed","banana",
"donut","skis","person","truck","bottle","suitcase","zebra","background")

val_class = ("umbrella","cow","cup","bus","keyboard","skateboard",
"dog","couch","tie","snowboard","sink","elephant","cake","scissors",
"airplane","cat","knife")

data_root = 'data/coco/'

data = dict(
    train=dict(classes=train_class),
    val=dict(classes=val_class, ann_file=data_root + 'annotations/val17.json'),
    test=dict(classes=val_class, ann_file=data_root + 'annotations/val17.json'))

# lr set for 2 gpu training
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
