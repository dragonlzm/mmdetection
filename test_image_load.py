import mmcv
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip

'''
def _convert_image_to_rgb(image):
    return image.convert("RGB")

file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)

filenname = '/data2/lwll/zhuoming/detection/coco/val2017/000000581100.jpg'
#filenname = 'CLIP.png'
img_bytes = file_client.get(filenname)
#img = mmcv.imfrombytes(img_bytes, flag='color')
img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')


print(img.shape)
print(type(img))
print(img[0])

# save on tensor before the conversion
img_tensor_original = torch.from_numpy(img)

# convert the numpy img to the PIL image 
PIL_image = Image.fromarray(np.uint8(img))

# convert to tensor
transform = Compose([
        _convert_image_to_rgb,
        ToTensor(),
    ])
img_tensor_original = torch.permute(img_tensor_original, (2, 0, 1))

converted_img_tensor = transform(PIL_image)
test_tensor = converted_img_tensor * 255
result = (test_tensor == img_tensor_original)
print(result)'''

# the result directly obtrained from CLIP code
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)

#filenname = '/data2/lwll/zhuoming/detection/coco/val2017/000000581100.jpg'
filenname = 'CLIP.png'
img_bytes = file_client.get(filenname)
#img = mmcv.imfrombytes(img_bytes, flag='color')
img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')



# convert the numpy img to the PIL image 
PIL_image = Image.fromarray(np.uint8(img))

# convert to tensor
transform = Compose([
        _convert_image_to_rgb,
        ToTensor(),
    ])

converted_img_tensor = transform(PIL_image)

clip_loaded_img = transform(Image.open("CLIP.png"))
