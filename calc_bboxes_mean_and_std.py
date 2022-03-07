import json
import numpy as np

test = json.load(open('/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json'))

all_w = []
all_h = []

for anno in test['annotations']:
    bbox = anno['bbox']
    w = bbox[2]
    h = bbox[3]
    all_w.append(w)
    all_h.append(h)
    
print('w_mean:', np.mean(all_w), 'h_mean:', np.mean(all_h))
print('w_std:', np.std(all_w), 'h_std:', np.std(all_h))