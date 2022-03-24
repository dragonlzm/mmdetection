_base_ = './mask_rcnn_distill_r50_fpn_1x_coco_2gpu.py'

classes = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=17,
                       fg_vec_cfg=dict(fixed_param=True, 
                                       load_path='/data2/lwll/zhuoming/detection/embeddings/base_finetuned_17cates.pt')),
                                       #load_path='/project/nevatia_174/zhuoming/detection/embeddings/base_finetuned_17cates.pt')),
        mask_head=dict(num_classes=48)))

#evaluation = dict(interval=1, metric=['bbox', 'segm'])