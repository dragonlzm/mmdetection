import json

imagenet_content = json.load(open('imagenet_hierachy.json'))

content_list = [imagenet_content]
all_names = []
### if now checking ele has the 'children', put all the children in the queue
while len(content_list) != 0:
    now_ele = content_list.pop()
    name = now_ele['name']
    all_names.append(name)
    if 'children' in now_ele:
        for child in now_ele['children']:
            content_list.append(child)

all_names = list(set(all_names))
print(len(all_names))

all_names.remove("ImageNet 2011 Fall Release")
clean_name = []
for name in all_names:
    if ',' in name:
        name = name.split(',')[0]
    clean_name.append(name)
