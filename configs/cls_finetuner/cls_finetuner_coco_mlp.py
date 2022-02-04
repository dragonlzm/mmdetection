_base_ = './cls_finetuner_coco.py'

# model settings
model = dict(
    rpn_head=dict(
        word_embeddings_path=None,
        linear_probe=False,
        mlp_probe=True)
)
