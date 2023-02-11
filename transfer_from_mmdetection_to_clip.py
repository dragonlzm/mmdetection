import torch

# original file path
mmdet_file_path = "/data/zhuoming/detection/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth"

mmdet_model = torch.load(mmdet_file_path)

new_file = {}
for name in mmdet_model['state_dict']:
    value = mmdet_model['state_dict'][name]
    # for the vision part
    if name.startswith('backbone'):
        new_name = '.'.join(['visual',] + name.split('.')[1:])
        new_file[new_name] = value
    # for the language part
    elif name.startswith('rpn_head'):
        new_name = '.'.join(name.split('.')[1:])
        new_file[new_name] = value
    else:
        print('exception name:', name)
    
torch.save(new_file, '/data/zhuoming/detection/pretrain/base48_finetuned_clip_vitb32_full.pth')