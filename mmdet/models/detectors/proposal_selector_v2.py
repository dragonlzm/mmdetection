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

    def forward_mask(self, gt_bboxes, gt_labels, proposal_bboxes, proposal_scores, proposal_feats=None):
        return pred_score, pred_score_target

    def forward_train(self,
                    img,
                    img_metas,
                    gt_bboxes=None,
                    gt_labels=None, 
                    proposal_bboxes=None,
                    proposal_clip_score=None,
                    clip_mask=None):
        print('img_metas', img_metas['img_shape'], 'clip_mask', clip_mask.shape)
        proposal_mask = self.convert_bboxes_to_mask(img_metas, proposal_bboxes, proposal_clip_score)
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
