_base_ = './rpn_attention-rpn_r50_c4_2xb4_coco_official-base-training.py'

classes = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

# for 2 gpu setting (2*4, maintaining the overall batchsize the same)
data = dict(
    train=dict(
        dataset=dict(classes=classes)),
    val=dict(classes=classes),
    test=dict(classes=classes),
    # random sample 10 shot base instance to evaluate training
    model_init=dict(classes=classes))