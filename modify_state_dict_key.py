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