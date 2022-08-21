import json

lvis_ann_file = "/data/zhuoming/detection/lvis_v1/annotations/lvis_v1_train.json"
lvis_ann_content = json.load(open(lvis_ann_file))

from_freq_to_cate_name = {}
for cate_info in lvis_ann_content['categories']:
    freq = cate_info['frequency']
    name = cate_info['name']
    if freq not in from_freq_to_cate_name:
        from_freq_to_cate_name[freq] = []
    from_freq_to_cate_name[freq].append(name)

print(from_freq_to_cate_name.keys())