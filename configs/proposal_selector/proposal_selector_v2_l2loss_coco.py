_base_ = './proposal_selector_v2_coco.py'

model = dict(
    loss=dict(type='MSELoss'))