import json
import torch


json_content = json.load(open('results.gt_acc.json'))

confusion_mat = torch.zeros([80,80])

for ele in json_content:
    res = ele['score']
    for pred, gt in zip(res[0], res[1]):
        confusion_mat[gt][pred] += 1

normalized_mat = confusion_mat / confusion_mat.sum(dim=-1, keepdim=True)
print(normalized_mat)

import pandas as pd
import numpy as np

temp = normalized_mat.numpy()
temp = pd.DataFrame(temp)
temp.to_csv('normalized_mat.csv')