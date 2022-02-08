_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ("umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife")

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

model = dict(
    rpn_head=dict(
        cate_names=["umbrella","cow","cup","bus","keyboard","skateboard",
            "dog","couch","tie","snowboard","sink","elephant","cake",
            "scissors","airplane","cat","knife"]    
        ))