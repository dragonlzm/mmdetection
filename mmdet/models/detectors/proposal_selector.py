# Copyright (c) OpenMMLab. All rights reserved.
from cgi import test
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

@DETECTORS.register_module()
class ProposalSelector(BaseDetector):
    """Implementation of Region Proposal Network."""

    def __init__(self,
                 encoder,
                 loss,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(ProposalSelector, self).__init__(init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = self.encoder.embed_dims
        self.linear_layer = Linear(self.embed_dims, 4)
        self.loss = build_loss(loss)
        self.iou_calculator = BboxOverlaps2D()
    
    def extract_feat(self):
        pass
    
    def forward_dummy(self, img):
        """Dummy forward function."""
        x = self.extract_feat(img)
        rpn_outs = self.rpn_head(x)
        return rpn_outs

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
            max_iou_per_proposal, _ = torch.max(real_iou, dim=0)
        all_pred_target.append(max_iou_per_proposal)
        
        return all_pred_target

    def forward_train(self,
                      gt_bboxes,
                      gt_labels, 
                      proposal_bboxes,
                      proposal_scores,
                      img_metas=None):
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
        # concate the proposal_bboxes and proposal_bboxes
        all_inputs = [torch.cat([proposal_bbox_per_img, proposal_score_per_img], dim=-1) 
                      for proposal_bbox_per_img, proposal_score_per_img in 
                      zip(proposal_bboxes, proposal_scores)]
        all_inputs = torch.cat(all_inputs, dim=0)
        
        memory = self.encoder(
            query=all_inputs,
            key=None,
            value=None,
            query_pos=None,
            query_key_padding_mask=None)
        pred_score = self.linear_layer(memory)
        
        pred_score_target = self.find_the_gt_for_proposal(proposal_bboxes, gt_bboxes)
        pred_score_target = torch.cat(pred_score_target, dim=0)
        
        loss_value = self.loss(pred_score, pred_score_target)
        loss_dict = dict()
        loss_dict['loss_value'] = loss_value
        return loss_dict

    def simple_test(self,
                    gt_bboxes,
                    gt_labels, 
                    proposal_bboxes,
                    proposal_scores,
                    img_metas=None):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[np.ndarray]: proposals
        """
        all_inputs = [torch.cat([proposal_bbox_per_img, proposal_score_per_img], dim=-1) 
                      for proposal_bbox_per_img, proposal_score_per_img in 
                      zip(proposal_bboxes, proposal_scores)]
        all_inputs = torch.cat(all_inputs, dim=0)
        
        memory = self.encoder(
            query=all_inputs,
            key=None,
            value=None,
            query_pos=None,
            query_key_padding_mask=None)
        pred_score = self.linear_layer(memory)
        
        pred_score_target = self.find_the_gt_for_proposal(proposal_bboxes, gt_bboxes)
        pred_score_target = torch.cat(pred_score_target, dim=0)
        
        # concat all the result, send them back to dataset and do the evaluation
        result = torch.cat([pred_score.unsqueeze(dim=0), pred_score_target.unsqueeze(dim=0)], dim=0)

        return [result.cpu().numpy()]

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