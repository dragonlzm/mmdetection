file1 = open("image_name.txt","r")
res = file1.readlines()
all_cate_names = []

for ele in res:
    raw_for_one_cate = ele.split('|')[2].strip().split(',')
    all_name_for_one_cate = []
    for name in raw_for_one_cate:
        if name == "":
            continue
        name = name.strip().strip("'").strip('"')
        all_name_for_one_cate.append(name)
    all_cate_names.append(all_name_for_one_cate)

print(all_cate_names)