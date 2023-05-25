_base_ = './proposal_selector_v2_coco.py'

model = dict(
    loss=dict(type='MSELoss'),
    ranking_loss='TripletMarginLoss',
    ranking_loss_only=True)

evaluation = dict(interval=3, metric='proposal_selection')