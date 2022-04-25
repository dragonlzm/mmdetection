import torch

semi_model = torch.load('/data/zhuoming/detection/test/mask_rcnn_clip_classifier/epoch_0.pth')

# roi_head.bbox_head.fc_cls_fg.weight
semi_model['roi_head.bbox_head.fc_cls_fg.weight'] = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_17cates.pt')


# roi_head.clip_backbone
# checkpoint="data/test/cls_finetuner_clip_base_all_train/epoch_12.pth", prefix='backbone.'
clip_finetuned_model = torch.load('/data/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth')['state_dict']
for name in semi_model:
    if name.startswith('roi_head.clip_backbone'):
        need_name = ['backbone'] + name.split('.')[2:]
        need_name = '.'.join(need_name)
        print('semi_model[name]', semi_model[name].shape, 'clip_finetuned_model[need_name]', clip_finetuned_model[need_name].shape)
        semi_model[name] = clip_finetuned_model[need_name]


torch.save(semi_model, '/data/zhuoming/detection/test/mask_rcnn_clip_classifier/novel17.pth')