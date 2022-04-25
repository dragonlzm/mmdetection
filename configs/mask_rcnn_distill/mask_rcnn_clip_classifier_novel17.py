_base_ = './mask_rcnn_clip_classifier.py'

classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes, eval_filter_empty_gt=True),
    test=dict(classes=classes, eval_filter_empty_gt=True))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=17,
                       fg_vec_cfg=dict(fixed_param=True, 
                                       #load_path='/data/zhuoming/detection/embeddings/base_finetuned_17cates.pt')),
                                       #load_path='/data2/lwll/zhuoming/detection/embeddings/base_finetuned_17cates.pt')),
                                       load_path='data/embeddings/base_finetuned_17cates.pt')),
        mask_head=dict(num_classes=17)),
    test_cfg=dict(
        #rpn=dict(
        #    nms_pre=1000,
        #    max_per_img=1000,
        #    nms=dict(type='nms', iou_threshold=0.7),
        #    min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)))

#evaluation = dict(interval=1, metric=['bbox', 'segm'])