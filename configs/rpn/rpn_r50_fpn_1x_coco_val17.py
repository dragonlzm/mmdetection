_base_ = './rpn_r50_fpn_1x_coco.py'

val_class = ("umbrella","cow","cup","bus","keyboard","skateboard",
"dog","couch","tie","snowboard","sink","elephant","cake","scissors",
"airplane","cat","knife")

data = dict(
    val=dict(classes=val_class),
    test=dict(classes=val_class))
