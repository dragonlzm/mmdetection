import json

json_content = json.load(open('/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'))

from_id_to_name = {ele['id']:ele['name'] for ele in json_content['categories']}
from_name_to_id = {ele['name']:ele['id'] for ele in json_content['categories']}


#cate_name = ["umbrella","cow","cup","bus","keyboard","skateboard","dog","couch","tie","snowboard","sink","elephant","cake","scissors","airplane","cat","knife"]

#cate_name = ["toilet","bicycle","apple","train","laptop","carrot","motorcycle","oven","chair","mouse","boat","kite","sheep","horse","sandwich","clock","tv","backpack","toaster","bowl","microwave","bench","book","orange","bird","pizza","fork","frisbee","bear","vase","toothbrush","spoon","giraffe","handbag","broccoli","refrigerator","remote","surfboard","car","bed","banana","donut","skis","person","truck","bottle","suitcase","zebra"]
cate_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush']
cate_id = [from_name_to_id[ele] for ele in cate_name]

cate_id.sort()

print(cate_id)

sorted_name = [from_id_to_name[ele] for ele in cate_id]