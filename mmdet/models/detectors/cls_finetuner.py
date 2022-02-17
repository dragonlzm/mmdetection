# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import bbox_mapping
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from PIL import Image
import numpy as np
import math
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


@DETECTORS.register_module()
class ClsFinetuner(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None,
                 init_cfg=None):
        super(ClsFinetuner, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck) if neck is not None else None
        #rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_train_cfg = train_cfg.get('rpn_head', None) if train_cfg is not None else None
        rpn_test_cfg = test_cfg.get('rpn_head', None) if test_cfg is not None else None
        rpn_head.update(train_cfg=rpn_train_cfg)
        rpn_head.update(test_cfg=rpn_test_cfg)
        self.rpn_head = build_head(rpn_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.preprocess = _transform(self.backbone.input_resolution)

        # deal with the crop size and location
        self.test_crop_size_modi_ratio = self.test_cfg.get('crop_size_modi', 1.2) if self.test_cfg is not None else 1.2
        self.test_crop_loca_modi_ratio = self.test_cfg.get('crop_loca_modi', 0) if self.test_cfg is not None else 0

        self.train_crop_size_modi_ratio = self.train_cfg.get('crop_size_modi', 1.2) if self.train_cfg is not None else 1.2
        self.train_crop_loca_modi_ratio = self.train_cfg.get('crop_loca_modi', 0) if self.train_cfg is not None else 0        

    def crop_img_to_patches(self, imgs, gt_bboxes, img_metas):
        # handle the test config
        if self.training: 
            crop_size_modi_ratio = self.train_crop_size_modi_ratio
            crop_loca_modi_ratio = self.train_crop_loca_modi_ratio
        else:
            crop_size_modi_ratio = self.test_crop_size_modi_ratio
            crop_loca_modi_ratio = self.test_crop_loca_modi_ratio            
        
        bs, c, _, _ = imgs.shape
        #'img_shape':  torch.Size([2, 3, 800, 1184])
        # what we need is [800, 1184, 3]
        imgs = imgs.permute(0, 2, 3, 1).numpy()

        all_results = []
        for img_idx in range(bs):
            H, W, channel = img_metas[img_idx]['img_shape']
            all_gt_bboxes = gt_bboxes[img_idx]
            if len(all_gt_bboxes) == 0:
                continue
            img = imgs[img_idx]
            result = []
            for box_i, bbox in enumerate(all_gt_bboxes):
                # the original bbox location
                tl_x, tl_y, br_x, br_y = bbox[0], bbox[1], bbox[2], bbox[3]
                x = tl_x
                y = tl_y
                w = br_x - tl_x
                h = br_y - tl_y
                # change the bbox location by changing the top left position
                # bbox change direction
                x_direction_sign = random.randint(-1,1)
                y_direction_sign = random.randint(-1,1)
                # bbox direction change ratio(the ration should be 1/2, 1/3, 1/4, 1/5)
                # commonly we will mantain the size of the bbox unchange while changing
                # the localization of the bbox
                x_change_pixel = w * crop_loca_modi_ratio * x_direction_sign
                y_change_pixel = h * crop_loca_modi_ratio * y_direction_sign

                # change the bbox size ratio
                x_change_for_size = ((crop_size_modi_ratio - 1) / 2) * w
                y_change_for_size = ((crop_size_modi_ratio - 1) / 2) * h

                # the final format for the
                x_start_pos = math.floor(max(x-x_change_for_size+x_change_pixel , 0))
                y_start_pos = math.floor(max(y-y_change_for_size+y_change_pixel, 0))
                x_end_pos = math.ceil(min(x+x_change_for_size+w, W-1))
                y_end_pos = math.ceil(min(y+y_change_for_size+h, H-1))

                #x_start_pos = math.floor(max(x-0.1*w, 0))
                #y_start_pos = math.floor(max(y-0.1*h, 0))
                #x_end_pos = math.ceil(min(x+1.1*w, W-1))
                #y_end_pos = math.ceil(min(y+1.1*h, H-1))

                now_patch = img[y_start_pos: y_end_pos, x_start_pos: x_end_pos, :]           
                # crop the GT bbox and place it in the center of the zero square
                gt_h, gt_w, c = now_patch.shape
                if gt_h != gt_w:
                    long_edge = max((gt_h, gt_w))
                    empty_patch = np.zeros((long_edge, long_edge, 3))
                    if gt_h > gt_w:
                        x_start = (long_edge - gt_w) // 2
                        x_end = x_start + gt_w
                        empty_patch[:, x_start: x_end] = now_patch
                    else:
                        y_start = (long_edge - gt_h) // 2
                        y_end = y_start + gt_h
                        empty_patch[y_start: y_end] = now_patch
                    now_patch = empty_patch
                
                #data = Image.fromarray(np.uint8(now_patch))
                #data.save('/data2/lwll/zhuoming/detection/test/cls_finetuner_clip_base_100shots_train/patch_visualize/' + img_metas[img_idx]['ori_filename'] + '_' + str(box_i) + '.png')
                #new_patch, w_scale, h_scale = mmcv.imresize(now_patch, (224, 224), return_scale=True)
                # convert the numpy to PIL image
                PIL_image = Image.fromarray(np.uint8(now_patch))
                # do the preprocessing
                new_patch = self.preprocess(PIL_image)
                #image_result.append(np.expand_dims(new_patch, axis=0))
                result.append(new_patch.unsqueeze(dim=0))
            result = torch.cat(result, dim=0)
            all_results.append(result)

        #cropped_patches = np.concatenate(result, axis=0)
        # the shape of the cropped_patches: torch.Size([gt_num_in_batch, 3, 224, 224])
        #cropped_patches = torch.cat(result, dim=0).cuda()
        return all_results

    def extract_feat(self, img, gt_bboxes, img_metas=None):
        """Extract features.

        Args:
            img (torch.Tensor): Image tensor with shape (n, c, h ,w).

        Returns:
            list[torch.Tensor]: Multi-level features that may have
                different resolutions.
        """
        # the image shape
        # it pad the images in the same batch into the same shape
        #torch.Size([2, 3, 800, 1088])
        #torch.Size([2, 3, 800, 1216])
        #torch.Size([2, 3, 800, 1216])
        #bs = img.shape[0]
        
        # crop the img into the patches with normalization and reshape
        # (a function to convert the img)
        #cropped_patches_list:len = batch_size, list[tensor] each tensor shape [gt_num_of_image, 3, 224, 224]
        cropped_patches_list = self.crop_img_to_patches(img.cpu(), gt_bboxes, img_metas)

        # convert dimension from [bs, 64, 3, 224, 224] to [bs*64, 3, 224, 224]
        #converted_img_patches = converted_img_patches.view(bs, -1, self.backbone.input_resolution, self.backbone.input_resolution)

        # the input of the vision transformer should be torch.Size([64, 3, 224, 224])
        result_list = []
        for patches in cropped_patches_list:
            x = self.backbone(patches.cuda())
            if self.with_neck:
                x = self.neck(x)
            result_list.append(x)
        # convert the feature [bs*64, 512] to [bs, 64, 512]
        #x = x.view(bs, -1, x.shape[-1])
        return result_list

    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      gt_labels=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        #if (isinstance(self.train_cfg.rpn, dict)
        #        and self.train_cfg.rpn.get('debug', False)):
        #    self.rpn_head.debug_imgs = tensor2imgs(img)

        x = self.extract_feat(img, gt_bboxes, img_metas)
        # x: list[tensor] each tensor shape [gt_num_of_image, 512]
        losses = self.rpn_head.forward_train(x, img_metas, gt_bboxes, gt_labels,
                                             gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, gt_bboxes, gt_labels, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        img = img.unsqueeze(dim=0)
        img_metas = [img_metas]

        x = self.extract_feat(img, gt_bboxes, img_metas)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        proposal_list = self.rpn_head.simple_test_bboxes(x, gt_labels, img_metas, gt_bboxes)
        #if rescale:
        #    for proposals, meta in zip(proposal_list, img_metas):
        #        proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])
        if torch.onnx.is_in_onnx_export():
            return proposal_list

        return [proposal.cpu().numpy() for proposal in proposal_list]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        proposal_list = self.rpn_head.aug_test_rpn(
            self.extract_feats(imgs), img_metas)
        if not rescale:
            for proposals, img_meta in zip(proposal_list, img_metas[0]):
                img_shape = img_meta['img_shape']
                scale_factor = img_meta['scale_factor']
                flip = img_meta['flip']
                flip_direction = img_meta['flip_direction']
                proposals[:, :4] = bbox_mapping(proposals[:, :4], img_shape,
                                                scale_factor, flip,
                                                flip_direction)
        return [proposal.cpu().numpy() for proposal in proposal_list]

    def show_result(self, data, result, top_k=20, **kwargs):
        """Show RPN proposals on the image.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            top_k (int): Plot the first k bboxes only
               if set positive. Default: 20

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        mmcv.imshow_bboxes(data, result, top_k=top_k, **kwargs)
