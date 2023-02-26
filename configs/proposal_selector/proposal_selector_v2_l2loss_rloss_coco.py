_base_ = './proposal_selector_v2_coco.py'

model = dict(
    loss=dict(type='MSELoss'),
    ranking_loss='TripletMarginLoss')

evaluation = dict(interval=3, metric='proposal_selection')