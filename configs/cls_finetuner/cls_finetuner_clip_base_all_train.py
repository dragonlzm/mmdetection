_base_ = './cls_finetuner_clip_full_coco.py'

data_root = 'data/coco/'

classes = ("toilet","bicycle","apple","train","laptop","carrot","motorcycle","oven","chair","mouse","boat","kite","sheep","horse","sandwich","clock","tv","backpack","toaster","bowl","microwave","bench","book","orange","bird","pizza","fork","frisbee","bear","vase","toothbrush","spoon","giraffe","handbag","broccoli","refrigerator","remote","surfboard","car","bed","banana","donut","skis","person","truck","bottle","suitcase","zebra","background")

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

#["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]
