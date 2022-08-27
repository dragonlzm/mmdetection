# Copyright (c) OpenMMLab. All rights reserved.
from hashlib import new
import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .standard_roi_head_distillation import StandardRoIHeadDistill
from .test_mixins import BBoxTestMixin, MaskTestMixin
import os
import json


@HEADS.register_module()
class StandardRoIHeadDistillWithTransformer(StandardRoIHeadDistill):
    """Simplest base roi head including one bbox head and one mask head."""

    def _bbox_forward(self, x, rois, distilled_feat=None, gt_rand_rois=None, gt_labels=None, img_metas=None, distill_ele_weight=None, bboxes_num=None):
        """Box head forward function used in both training and testing.
        bboxes_num: list[tuple(gt_bbox_num, rand_bbox_num, proposal_number)] or list[tuple(proposal_number, )]
        """  
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # is the number of feat map layer
        if distilled_feat != None and gt_rand_rois != None:
            gt_and_rand_bbox_feat = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], gt_rand_rois)  
        
        # we use the fpn do not need to consider the share head
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            if distilled_feat != None and gt_rand_rois != None:
                gt_and_rand_bbox_feat = self.shared_head(gt_and_rand_bbox_feat)  
        
        # if training
        if distilled_feat != None and gt_rand_rois != None:
            cls_score, bbox_pred, gt_and_bg_feats = self.bbox_head(bbox_feats, rois, img_metas, gt_rand_rois=gt_rand_rois,
                                                                   gt_and_rand_bbox_feat=gt_and_rand_bbox_feat, bboxes_num=bboxes_num)
        else:
            cls_score, bbox_pred, gt_and_bg_feats = self.bbox_head(bbox_feats, rois, img_metas, bboxes_num=bboxes_num)
        

        # send the features and the bbox into the transformer for each image in the batch
        
        # split the feature for distillation and for the final prediction
        
        # calcualate the distillation loss

        

        
        # if we use bg proposal, the cls_score will has the the length of samples + bg number
        
        # for save the classification feat
        if self.save_the_feat is not None:
            if not os.path.exists(self.save_the_feat):
                os.makedirs(self.save_the_feat)
            file_name = img_metas[0]['ori_filename'].split('.')[0] + '.json'
            random_file_path = os.path.join(self.save_the_feat, file_name)  
            file = open(random_file_path, 'w')
            result_json = {'feat':gt_and_bg_feats.cpu().tolist()}
            file.write(json.dumps(result_json))
            file.close()
        
        # obtain the feat for the distillation
        if distilled_feat != None and gt_rand_rois != None:
            _, _, pred_feats = self.bbox_head(gt_and_rand_bbox_feat)
            # normalize the distilled feat
            cat_distilled_feat = torch.cat(distilled_feat, dim=0)
            if distill_ele_weight:
                distill_ele_weight = torch.cat(distill_ele_weight, dim=0)
            cat_distilled_feat = cat_distilled_feat / cat_distilled_feat.norm(dim=-1, keepdim=True)
            
            distill_loss_value = self.distillation_loss(pred_feats, cat_distilled_feat, distill_ele_weight)
            distill_loss_value *= (self.bbox_head.clip_dim * self.distill_loss_factor)

        # if training
        if distilled_feat != None and gt_rand_rois != None:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, distill_loss_value=dict(distill_loss_value=distill_loss_value))
        else:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
