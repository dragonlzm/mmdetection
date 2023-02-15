# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
from pickle import FALSE
import warnings

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import bbox_mapping
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D
from ..builder import DETECTORS, build_backbone, build_head, build_neck, HEADS, build_loss
from .base import BaseDetector
from PIL import Image
import numpy as np
import math
import random
import os
import json
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmdet.core.bbox.iou_calculators.iou2d_calculator import BboxOverlaps2D


def get_points_single(featmap_size, dtype, device):
    """Get points of a single scale level."""
    h, w = featmap_size
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    x_range = torch.arange(w, device=device).to(dtype)
    y_range = torch.arange(h, device=device).to(dtype)
    y, x = torch.meshgrid(y_range, x_range)
    points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
    return points


def _get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    """Compute regression and classification targets for a single image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), num_classes), \
                gt_bboxes.new_zeros((num_points, 4))
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
    xs, ys = points[:, 0], points[:, 1]
    xs = xs[:, None].expand(num_points, num_gts)
    ys = ys[:, None].expand(num_points, num_gts)
    left = xs - gt_bboxes[..., 0]
    right = gt_bboxes[..., 2] - xs
    top = ys - gt_bboxes[..., 1]
    bottom = gt_bboxes[..., 3] - ys
    bbox_targets = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    return inside_gt_bbox_mask


def get_target_single(gt_bboxes, gt_labels, points, num_classes=65):
    # this function return the per categories mask
    num_points = points.size(0)
    areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
    gt_bboxes[:, 3] - gt_bboxes[:, 1])
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    
    total_iteration = 10
    gap_per_iter = gt_bboxes.shape[0] / total_iteration
    all_result = []
    for i in range(total_iteration):
        start = int(i * gap_per_iter)
        end = int((i+1) * gap_per_iter)
        #print('start:end', start, end)
        _inside_gt_bbox_mask = _get_target_single(gt_bboxes[start:end], gt_labels[start:end], all_points)
        #print(temp_assigned_label.shape)
        all_result.append(_inside_gt_bbox_mask)
    # inside_gt_bbox_mask will be a tensor([num_of_pixel, num_of_proposal]), a true/ false mask
    inside_gt_bbox_mask = torch.cat(all_result, dim=-1)
    inside_gt_bbox_mask = inside_gt_bbox_mask.permute([1,0])
    #print(inside_gt_bbox_mask)
    
    mask_of_all_cates = []
    for i in range(num_classes):
        cate_matched_idx = (gt_labels == i)
        # selected_masks tensor(num_of_mask, num_of_pixel)
        selected_masks = inside_gt_bbox_mask[cate_matched_idx]
        mask_per_cate = torch.sum(selected_masks, dim=0)
        mask_of_all_cates.append(mask_per_cate.unsqueeze(dim=0))

    mask_of_all_cates = torch.cat(mask_of_all_cates, dim=0)
    return mask_of_all_cates    


@DETECTORS.register_module()
class ProposalSelectorV2(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 encoder,
                 loss,
                 input_dim=5,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(ProposalSelectorV2, self).__init__(init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = self.encoder.embed_dims
        self.input_dim = input_dim
        self.input_proj = Linear(self.input_dim, self.embed_dims)
        self.linear_layer = Linear(self.embed_dims, 1)
        self.loss = build_loss(loss)
        self.iou_calculator = BboxOverlaps2D()
    
    def extract_feat(self):
        pass
    
    def forward_dummy(self, img):
        pass

    def find_the_gt_for_proposal(self, 
                                 proposal_bboxes, 
                                 gt_bboxes):
        '''
        the proposal_bboxes (list[Tensor])
        the gt_bboxes (list[Tensor])
        '''
        
        all_pred_target = []
        for proposal, gt_bbox in zip(proposal_bboxes, gt_bboxes):
            real_iou = self.iou_calculator(proposal, gt_bbox)
            max_iou_per_proposal, _ = torch.max(real_iou, dim=1)
            all_pred_target.append(max_iou_per_proposal)
        
        return all_pred_target

    def convert_bboxes_to_mask(self, proposal_bboxes, pred_distri, image_size):
        '''proposal_bboxes should be [1000, 4], the pred_distri should be [1000, 65]
           The output should be [1000, H, W, 65]'''

    def convert_bboxes_to_mask(self, proposal_bboxes, proposal_clip_score):
        pass
    
    def forward_mask(self, gt_bboxes, gt_labels, proposal_bboxes, proposal_scores, proposal_feats=None):
        pass
    
    def generate_per_img_mask(self, img_metas, proposal_bboxes, proposal_clip_score):
        all_points = get_points_single((h,w), torch.float16, torch.device('cuda'))
        mask_of_all_cates = get_target_single(predict_boxes, predicted_labels, all_points)

    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None, 
                    proposal_bboxes=None,
                    proposal_clip_score=None,
                    clip_mask=None):
        print('img_metas', img_metas['img_shape'], 'clip_mask', clip_mask.shape)
        per_cate_mask = self.generate_per_img_mask(img_metas, proposal_bboxes, proposal_clip_score)
        
        per_proposal_mask = self.convert_bboxes_to_mask(img_metas, proposal_bboxes, proposal_clip_score)
        pred_score, pred_score_target = self.forward_mask(gt_bboxes, gt_labels, proposal_bboxes, proposal_scores, proposal_feats)
        
        loss_value = self.loss(pred_score, pred_score_target)
        loss_dict = dict()
        loss_dict['loss_value'] = loss_value
        return loss_dict

    def simple_test(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None, 
                    proposal_bboxes=None,
                    proposal_scores=None,
                    proposal_feats=None,
                    rescale=False):

        pred_score, pred_score_target = self.forward_mask(gt_bboxes, gt_labels, proposal_bboxes, proposal_scores, proposal_feats)
        
        # concat all the result, send them back to dataset and do the evaluation
        result = torch.cat([pred_score.unsqueeze(dim=0), pred_score_target.unsqueeze(dim=0)], dim=0)

        return [result.cpu().numpy()]

    def aug_test(self, imgs, img_metas, rescale=False):
        pass
    
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
