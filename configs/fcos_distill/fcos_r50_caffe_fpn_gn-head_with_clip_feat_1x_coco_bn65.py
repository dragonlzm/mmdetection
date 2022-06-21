_base_ = './fcos_r50_caffe_fpn_gn-head_with_clip_feat_1x_coco.py'

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    bbox_head=dict(num_classes=65,
                    fg_vec_cfg=dict(fixed_param=True, 
                                    load_path='data/embeddings/raw_65cates.pt')))