# this script aims to collect all the word from the coco caption dataset
import json
import re
file_path = '/data/zhuoming/detection/coco/annotations/captions_train2017.json'
file_content = json.load(open(file_path))

def fix_the_text(text):
    text = text.lower()
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text

def check_the_word(word):
    if '\n' in word or '"' in word or '/' in word or "'" in word or '.' in word or '(' in word or ')' in word or ',' in word or '\\' in word:
        return False
    else:
        return True

all_words = []
for ele in file_content['annotations']:
    now_cap = ele['caption']
    words_per_sentence = now_cap.split(' ')
    stripped_result = []
    for word in words_per_sentence:
        word = fix_the_text(word)
        if not check_the_word(word):
            continue
        stripped_result.append(word)
    all_words += stripped_result

all_words = list(set(all_words))
print(len(all_words))
file = open('all_words.json', 'w')
file.write(json.dumps(all_words))
file.close()


# prepare the word into phrases
all_phrases = ["a " + word for word in all_words]

# tokenize all the word and get the feature
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(all_phrases).to(device)

result_dict = {}
for name, encoded_text in zip(all_words, text):
    encoded_text = encoded_text.unsqueeze(dim=0)
    with torch.no_grad():
        text_features = model.encode_text(encoded_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        result_dict[name] = text_features.cpu().tolist()

file = open('phrases_vs_feats.json', 'w')
file.write(json.dumps(result_dict))
file.close()

# generate the feature for all coco categories
import json
all_cate_name = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# prepare the word into phrases
all_phrases = ["a " + word for word in all_cate_name]

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(all_phrases).to(device)

result_dict = {}
for name, encoded_text in zip(all_cate_name, text):
    encoded_text = encoded_text.unsqueeze(dim=0)
    with torch.no_grad():
        text_features = model.encode_text(encoded_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        result_dict[name] = text_features.cpu().tolist()

file = open('coco_names_vs_feats.json', 'w')
file.write(json.dumps(result_dict))
file.close()

# calculate the cosine similarity
import json
import torch

# cate_feat = json.load(open('coco_names_vs_feats.json'))
# phrase_feat = json.load(open('phrases_vs_feats.json'))
cate_feat = json.load(open('finetuned_coco_names_vs_feats.json'))
phrase_feat = json.load(open('finetuned_phrases_vs_feats.json'))

all_cate_name = []
all_cate_feat = []
for key in cate_feat:
    all_cate_name.append(key)
    all_cate_feat.append(torch.tensor(cate_feat[key]))

all_cate_feat = torch.cat(all_cate_feat, dim=0)

all_phrase_name = []
all_phrase_feat = []
for key in phrase_feat:
    all_phrase_name.append(key)
    all_phrase_feat.append(torch.tensor(phrase_feat[key]))

all_phrase_feat = torch.cat(all_phrase_feat, dim=0)

# calculate the cosine similarity 
cos_per_cate = all_cate_feat @ all_phrase_feat.t()
top_100_idx = torch.topk(cos_per_cate, 500, dim=-1)[1]

all_result = {}
for i, name in enumerate(all_cate_name):
    cate_idx = top_100_idx[i]
    all_top_100_similar_name = [all_phrase_name[idx] for idx in cate_idx if name not in all_phrase_name[idx]]
    all_result[name] = all_top_100_similar_name

#print(all_result)

#file = open('top_500_attribute.json', 'w')
file = open('finetuned_top_500_attribute.json', 'w')
file.write(json.dumps(all_result))
file.close()