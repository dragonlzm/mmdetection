# this script aims to calculate the cosine similarity between the embeddings

import torch

embeddings = torch.load('base_finetuned_17cates.pt')

result = []

for i, outer_vector in enumerate(embeddings):
    outer_vector /= outer_vector.norm(dim=-1, keepdim=True)
    result.append([])
    for j, inner_vector in enumerate(embeddings):
        inner_vector /= inner_vector.norm(dim=-1, keepdim=True)
        value = (outer_vector * inner_vector).sum().item()
        result[i].append(value)

print(result)
    