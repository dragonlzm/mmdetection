_base_ = './fcos_r50_caffe_fpn_gn-head_with_clip_feat_1x_coco.py'

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
    bbox_head=dict(num_classes=17,
                    fg_vec_cfg=dict(fixed_param=True, 
                                    load_path='data/embeddings/raw_17cates.pt')))