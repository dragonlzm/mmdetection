# from the clip model obtain the vision section only model state_dict

import os
import clip
import torch
from torchvision.datasets import CIFAR100
from collections import OrderedDict


# Load the model
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load('ViT-B/32', device)


new_state_dict = OrderedDict()
for param_name in model.state_dict():
    splited_name = param_name.split('.')
    # skip all the params that do not start with 'visual'
    if splited_name[0] != 'visual':
        continue
    new_name = '.'.join(splited_name[1:])
    new_state_dict[new_name] = model.state_dict()[param_name]
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())


torch.save(new_state_dict, './modified_state_dict.pth')


for param_tensor in new_state_dict:
    print(param_tensor, "\t", new_state_dict[param_tensor].size())

# Print optimizer's state_dict
torch.save(model.cpu().state_dict(), 'ViT_B32_full.pt')



import os
import clip
import torch
from torchvision.datasets import CIFAR100
from collections import OrderedDict

needed_name = ['stem.0.weight', 'stem.1.weight', 'stem.1.bias', 'stem.1.running_mean', 'stem.1.running_var', 'stem.3.weight', 'stem.4.weight', 'stem.4.bias', 'stem.4.running_mean', 'stem.4.running_var', 'stem.6.weight', 'stem.7.weight', 'stem.7.bias', 'stem.7.running_mean', 'stem.7.running_var']
now_name = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'conv2.weight', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'conv3.weight', 'bn3.weight', 'bn3.bias', 'bn3.running_mean', 'bn3.running_var', 'bn3.num_batches_tracked']

from_now_name_to_needed_name = {now:need for now, need in zip(now_name, needed_name)}

# Load the model
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
model, preprocess = clip.load("RN50", device)

model.cpu()
new_state_dict = OrderedDict()
for param_name in model.state_dict():
    print('param_name', param_name)
    split_name = '.'.join(param_name.split('.')[1:])
    print('split_name', split_name)
    if split_name in from_now_name_to_needed_name:
        new_name = 'visual.' +  from_now_name_to_needed_name[split_name]
        print('new_name', new_name)
        new_state_dict[new_name] = model.state_dict()[param_name]
    else:
        new_state_dict[param_name] = model.state_dict()[param_name]
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())

torch.save(new_state_dict, '/data/zhuoming/detection/pretrained/clip_rn50_full_name_modified.pth')