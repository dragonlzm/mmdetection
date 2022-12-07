_base_ = './cls_proposal_generator_coco.py'

model = dict(
    rpn_head=dict(
        num_classes=10,
        cate_names=['animal', 'plant', 'person', 'fungus', 'artifact', 'natural object', 'beverage', 'geological formation', 'nutriment', 'vegetable']))

