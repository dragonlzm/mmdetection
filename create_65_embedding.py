cate_name_48 = ['person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush']

cate_name_17 = ['airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors']

cate_name_65 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
 'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
 'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
 'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush']

import torch
#base_embedding = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_48cates.pt')
#novel_embedding = torch.load('/data/zhuoming/detection/embeddings/base_finetuned_17cates.pt')
base_embedding = torch.load('/data/zhuoming/detection/embeddings/raw_rn50_48cates.pt')
novel_embedding = torch.load('/data/zhuoming/detection/embeddings/raw_rn50_17cates.pt')


base_idx = 0
novel_idx = 0
sorted_embedding = []
for name in cate_name_65:
    if name in cate_name_48:
        sorted_embedding.append(base_embedding[base_idx].unsqueeze(dim=0))
        print('base', name, base_idx)
        base_idx += 1
    else:
        sorted_embedding.append(novel_embedding[novel_idx].unsqueeze(dim=0))
        print('novel', name, novel_idx)
        novel_idx += 1
        
sorted_embedding = torch.cat(sorted_embedding, dim=0)
#torch.save(sorted_embedding, '/data/zhuoming/detection/embeddings/base_finetuned_65cates.pt')
torch.save(sorted_embedding, '/data/zhuoming/detection/embeddings/raw_rn50_65cates.pt')