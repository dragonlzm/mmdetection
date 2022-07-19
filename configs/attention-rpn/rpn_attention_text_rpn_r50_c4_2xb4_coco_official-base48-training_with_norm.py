_base_ = './rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py'

# for 2 gpu setting (2*4, maintaining the overall batchsize the same)
model = dict(
    rpn_head=dict(
        normalize_img_feat=True
    )
)