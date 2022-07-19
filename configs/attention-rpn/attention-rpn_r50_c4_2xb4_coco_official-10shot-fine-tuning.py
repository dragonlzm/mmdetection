_base_ = './attention-rpn_r50_c4_4xb2_coco_official-10shot-fine-tuning.py'

# for 2 gpu setting (2*4, maintaining the overall batchsize the same)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4)

