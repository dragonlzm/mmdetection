_base_ = './mask_rcnn_with_base48_tuned_clip_feat_r50_caffe_c4_1x_coco_base48_200clip_pro_with_flip.py'

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
    roi_head=dict(
        bbox_head=dict(num_classes=65,
                       fg_vec_cfg=dict(fixed_param=True, 
                                       #load_path='/data2/lwll/zhuoming/detection/embeddings/base_finetuned_48cates.pt',
                                       load_path='data/embeddings/base_finetuned_65cates.pt')),
        mask_head=dict(num_classes=65)))