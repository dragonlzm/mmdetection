from numpy.lib.type_check import imag
from torch import tensor
import mmcv
import numpy as np
import json
import math
import os
#import clip
import torch
import io
from PIL import Image

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


# prepare the model
# Load the model
#device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load('ViT-B/32', device)

#json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_val2017.json'
json_file_path = '/data2/lwll/zhuoming/detection/coco/annotations/instances_train2017.json'

# load the json file
json_val = json.load(open(json_file_path))

# aggregate the annotation for each image
#file_root = '/data2/lwll/zhuoming/detection/coco/val2017/'
file_root = '/data2/lwll/zhuoming/detection/coco/train2017/'
from_img_id_to_bbox = {}

# go through 'images' first
for anno in json_val['images']:
    image_id = anno['id']
    if image_id not in from_img_id_to_bbox:
        from_img_id_to_bbox[image_id] = {'shape':[anno['height'] ,anno['width'] ,3], 'path': file_root + anno['file_name'], 'bbox':[]}

# go through the 'annotations'
for anno in json_val['annotations']:
    image_id = anno['image_id']
    box = anno['bbox']
    box.append(anno['category_id'])
    from_img_id_to_bbox[image_id]['bbox'].append(box)


#all_assigned_res = []
all_assigned_res = {}
all_feature_res = []
file_client_args=dict(backend='disk')
file_client = mmcv.FileClient(**file_client_args)


# go through all the image in the dict:
for count_i, image_id in enumerate(from_img_id_to_bbox.keys()):
    #filenname = from_img_id_to_bbox[image_id]['path']
    
    #load the image and convert to numpy
    #img_bytes = file_client.get(filenname)
    #img = mmcv.imfrombytes(img_bytes, flag='color', channel_order='rgb')
    
    h_patch_num = 8
    w_patch_num = 8
    #h_patch_num = 4
    #w_patch_num = 4
    
    #H1, W1, channel1 = img.shape
    H, W, channel = from_img_id_to_bbox[image_id]['shape']

    patch_H, patch_W = H / h_patch_num, W / w_patch_num
    h_pos = [int(patch_H) * i for i in range(h_patch_num + 1)]
    w_pos = [int(patch_W) * i for i in range(w_patch_num + 1)]

    # the grid match list
    match_list = [[[] for j in range(w_patch_num)] for i in range(h_patch_num)]

    for bbox in from_img_id_to_bbox[image_id]['bbox']:
        # for each bbox we need to calculate whether the bbox is inside the grid
        x, y, w, h, cat_id = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        x_idx = math.floor(x / patch_W)
        y_idx = math.floor(y / patch_H)
        x_end_pos = w_pos[x_idx+1]
        y_end_pos = h_pos[y_idx+1]
        br_x = x + w
        br_y = y + h
        # if the bbox is inside the grid
        # assign the bbox to the grid and save the categories
        if br_x <= x_end_pos and br_y <= y_end_pos:
            match_list[y_idx][x_idx].append([x, y, br_x, br_y, bbox[4]])
            continue
        # then we calculate the iou 
        # edge condition
        if br_x == math.floor(br_x / patch_W) * patch_W:
            br_x_idx = math.floor(br_x / patch_W) - 1
        else:
            br_x_idx = math.floor(br_x / patch_W)

        if br_y == math.floor(br_y / patch_H) * patch_H:
            br_y_idx = math.floor(br_y / patch_H) - 1
        else:    
            br_y_idx = math.floor(br_y / patch_H)

        # prepare the tensor for calculate the iou
        #the_gt_tensor = torch.tensor([[[x, y, br_x, br_y]]])
        # prepare the tensor for the grid
        #the_grid_list = []
        #for now_x_idx in range(x_idx, br_x_idx+1):
        #    for now_y_idx in range(y_idx, br_y_idx+1):
        #        now_grid = torch.tensor([[[w_pos[now_x_idx], h_pos[now_y_idx], w_pos[now_x_idx+1], h_pos[now_y_idx+1]]]])
        #        the_grid_list.append(now_grid)
        #the_grid_tensor = torch.cat(the_grid_list, dim=1)
        #assign_result = bbox_overlaps(the_gt_tensor, the_grid_tensor)
        #assign_result = bbox_overlaps(the_gt_tensor, the_grid_tensor, mode='iof')
        #result = assign_result >= 0.5
        #result = (assign_result == torch.max(assign_result))
        #print(assign_result)
        #break
        for now_x_idx in range(x_idx, br_x_idx+1):
            for now_y_idx in range(y_idx, br_y_idx+1):
                #list_idx = (now_y_idx - y_idx) * (br_x_idx+1 - x_idx) + (now_x_idx - x_idx)
                #if result[0][0][list_idx] == True:
                match_list[now_y_idx][now_x_idx].append([x, y, br_x, br_y, bbox[4]])

    if count_i % 1000 == 0:
        print(count_i)

    # the post-processing the final result should be a 8*8 tensor which each element is the categories id 
    final_assign_tensor = torch.zeros(h_patch_num, w_patch_num)
    for i in range(h_patch_num):
        for j in range(w_patch_num):
            if len(match_list[i][j]) == 0:
                final_assign_tensor[i][j] = 10000
            elif len(match_list[i][j]) == 1:
                final_assign_tensor[i][j] = match_list[i][j][0][-1]
            else:
                # the grid tensor 
                the_grid_tensor = torch.tensor([[[w_pos[j], h_pos[i], w_pos[j+1], h_pos[i+1]]]])
                # the possible match tensor
                all_tensor_list = []
                for bbox in match_list[i][j]:
                    now_tensor = torch.tensor([[bbox[:-1]]])
                    all_tensor_list.append(now_tensor)
                the_gt_tensor = torch.cat(all_tensor_list, dim=1)
                
                #calculate the iou between the gt bboxes and the 
                iou_value = bbox_overlaps(the_grid_tensor, the_gt_tensor, mode='iof')

                final_assign_tensor[i][j] = match_list[i][j][torch.argmax(iou_value)][-1]
    
    #all_assigned_res.append(torch.unsqueeze(final_assign_tensor, dim=0))
    all_assigned_res[image_id] = final_assign_tensor

    # crop the image and combine them together
    #result = []
    #for i in range(h_patch_num):
    #    h_start_pos = h_pos[i]
    #    h_end_pos = h_pos[i+1]
    #    for j in range(w_patch_num):
    #        w_start_pos = w_pos[j]
    #        w_end_pos = w_pos[j+1]
            # cropping the img into the patches which size is (H/8) * (W/8)
            # use the numpy to crop the image
    #        now_patch = img[h_start_pos: h_end_pos, w_start_pos: w_end_pos, :]
    #        PIL_image = Image.fromarray(np.uint8(now_patch))
            # do the preprocessing
    #        new_patch = preprocess(PIL_image)
    #        result.append(new_patch.unsqueeze(dim=0))

    #cropped_patches = torch.cat(result, dim=0).cuda()
    #with torch.no_grad():
    #    image_features = model.encode_image(cropped_patches)
    #all_feature_res.append(torch.unsqueeze(image_features, dim=0).cpu())

# Save to file
#all_assigned_res = torch.cat(all_assigned_res, dim=0)
#all_feature_res = torch.cat(all_feature_res, dim=0)

#np.save('assigned_res_4_by_4.npy', all_assigned_res.reshape(-1).numpy())
#torch.save(all_assigned_res, 'new_assigned_gt_4_by_4_train.pt')
torch.save(all_assigned_res, 'new_assigned_gt_8_by_8_train.pt')
#torch.save(all_feature_res, 'feature_4_by_4.pt')



