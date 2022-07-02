from statistics import mode
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN50", device=device)


torch.save(model.state_dict(), '/data/zhuoming/detection/pretrain/clip_rn50_full.pth')