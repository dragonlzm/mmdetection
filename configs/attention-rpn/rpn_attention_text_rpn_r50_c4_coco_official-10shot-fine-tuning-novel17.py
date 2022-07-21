_base_ = './rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py'

zero_shot_split = dict(
    ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
            'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush'),
    NOVEL_CLASSES=('airplane', 'bus', 'cat', 'dog', 'cow', 
            'elephant', 'umbrella', 'tie', 'snowboard', 
            'skateboard', 'cup', 'knife', 'cake', 'couch', 
            'keyboard', 'sink', 'scissors'),
    BASE_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush'))

data = dict(
    train=dict(
        dataset=dict(def_coco_split=zero_shot_split,
                     ann_cfg=[dict(method='Attention_RPN', setting='Novel17_10SHOT')])),
    val=dict(def_coco_split=zero_shot_split),
    test=dict(def_coco_split=zero_shot_split),
    # random sample 10 shot base instance to evaluate training
    model_init=dict(def_coco_split=zero_shot_split))

load_from = ('data/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training/'
             'latest.pth')
