# this script aims to generate the svd conversion matrix
import torch

test = torch.load('base_finetuned_65cates.pt')
U, S, Vh = torch.svd(test)
result = torch.mm(Vh[:, 1:], torch.transpose(Vh[:, 1:], 0, 1))
torch.save(result, 'conversion_mat.pt')