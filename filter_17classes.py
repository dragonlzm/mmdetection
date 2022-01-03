import json

val_class = ("umbrella","cow","cup","bus","keyboard","skateboard",
"dog","couch","tie","snowboard","sink","elephant","cake","scissors",
"airplane","cat","knife")

json_file = json.load(open('instances_val2017.json'))

from_name_to_id = {}

for anno in json_file['categories']:
    from_name_to_id[anno['name']] = anno['id']

all_cate_id = [from_name_to_id[ele] for ele in val_class]

print(all_cate_id)

filtered_annotation = []

for anno in json_file['annotations']:
    if anno['category_id'] in all_cate_id:
        filtered_annotation.append(anno)

final_anno = json_file
final_anno['annotations'] = filtered_annotation

file = open('val17.json', 'w')
file.write(json.dumps(final_anno))
file.close()


