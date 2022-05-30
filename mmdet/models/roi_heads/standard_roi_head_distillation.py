# Copyright (c) OpenMMLab. All rights reserved.
from hashlib import new
import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor, build_loss
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class StandardRoIHeadDistill(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        self.avg_pool = nn.AvgPool2d(self.bbox_head.roi_feat_size)
        self.distillation_loss_config = dict(type='L1Loss', loss_weight=1.0)
        self.distillation_loss = build_loss(self.distillation_loss_config)
        self.distill_loss_factor = self.train_cfg.get('distill_loss_factor', 1) if self.train_cfg is not None else 1
        self.match_count = 0
        self.total = 0

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      distilled_feat=None, 
                      rand_bboxes=None,
                      bg_bboxes=None,
                      bg_feats=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.
            distilled_feat (list[Tensor]): only contain the feat for the gt bboxes
                and the random bboxes

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, distilled_feat,
                                                    rand_bboxes, 
                                                    bg_bboxes=bg_bboxes,
                                                    bg_feats=bg_feats)
            losses.update(bbox_results['loss_bbox'])
            losses.update(bbox_results['distill_loss_value'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward(self, x, rois, distilled_feat=None, gt_rand_rois=None, gt_labels=None):
        """Box head forward function used in both training and testing."""  
        # is the number of feat map layer
        if distilled_feat != None and gt_rand_rois != None:
            # gt and random bbox feat from backbone_to
            # gt_and_rand_bbox_feat: torch.Size([1024, 256, 7, 7])
            gt_and_rand_bbox_feat = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], gt_rand_rois)
            # conduct the global averger pooling on the gt_and_rand_bbox_feat
            #gt_and_rand_bbox_feat = self.avg_pool(gt_and_rand_bbox_feat)
            # convert to shape from [221, 512, 1, 1] to [221, 512]
            #gt_and_rand_bbox_feat = gt_and_rand_bbox_feat.view(-1, self.bbox_roi_extractor.out_channels)
            # concatenate the distilled_feat
            
            # calculate the distill loss
            #distill_loss_value = self.distillation_loss(gt_and_rand_bbox_feat, distilled_feat)        
        
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # we use the fpn do not need to consider the share head
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
            gt_and_rand_bbox_feat = self.shared_head(gt_and_rand_bbox_feat)
        
        # if we use bg proposal, the cls_score will has the the length of samples + bg number
        cls_score, bbox_pred, gt_and_bg_feats = self.bbox_head(bbox_feats)
        
        # obtain the feat for the distillation
        if distilled_feat != None and gt_rand_rois != None:
            _, _, pred_feats = self.bbox_head(gt_and_rand_bbox_feat)
            # normalize the distilled feat
            cat_distilled_feat = torch.cat(distilled_feat, dim=0)
            cat_distilled_feat = cat_distilled_feat / cat_distilled_feat.norm(dim=-1, keepdim=True)
            distill_loss_value = self.distillation_loss(pred_feats, cat_distilled_feat)
            #distill_loss_value *= (self.bbox_head.clip_dim * 0.5)
            distill_loss_value *= (self.bbox_head.clip_dim * self.distill_loss_factor)
            
            '''
            # test the feat is matched or not
            gt_feat = [all_feats[:len(gt_lab)] for all_feats, gt_lab in zip(distilled_feat, gt_labels)]
            #print([ele.shape for ele in gt_feat])
            gt_feat = torch.cat(gt_feat, dim=0)
            gt_feat = gt_feat / gt_feat.norm(dim=-1, keepdim=True)
            # calculate the cos simiarity
            fg_score = self.bbox_head.fc_cls_fg(gt_feat)
            print('self.bbox_head.load_value.t()', self.bbox_head.load_value.t(), 'self.bbox_head.fc_cls_fg', self.bbox_head.fc_cls_fg)
            #fg_score = gt_feat @ self.bbox_head.load_value.t()
            #bg_score = self.fc_cls_bg(x_cls)
            # find the max cos value class
            max_id = torch.max(fg_score, dim=-1)[1]
            
            # calculate the acc
            cat_gt_label = torch.cat(gt_labels, dim=0)
            
            print('max_id', max_id, 'cat_gt_label', cat_gt_label)
            self.match_count += torch.sum((max_id == cat_gt_label)).item()
            self.total += max_id.shape[0]
            print('accumulated acc:', self.match_count / self.total)'''

        if distilled_feat != None and gt_rand_rois != None:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, distill_loss_value=dict(distill_loss_value=distill_loss_value))
        else:
            bbox_results = dict(
                cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, 
                            gt_bboxes, gt_labels,
                            img_metas, distilled_feat, 
                            rand_bboxes,
                            bg_bboxes=None,
                            bg_feats=None):
        """Run forward function and calculate loss for box head in training."""
        
        # prepare the roi for the proposal
        if self.use_bg_pro_as_ns:
            rois = bbox2roi([torch.cat([res.bboxes, bg_bbox]).cuda() for res, bg_bbox in zip(sampling_results, bg_bboxes)])
        else:     
            rois = bbox2roi([res.bboxes for res in sampling_results])
        # prepare the roi for the gt and the random bboxes
        if self.use_bg_pro_for_distill:
            gt_rand_rois = bbox2roi([torch.cat([gt_bbox, random_bbox, bg_bbox], dim=0) for gt_bbox, random_bbox, bg_bbox in zip(gt_bboxes, rand_bboxes, bg_bboxes)])
        else:   
            gt_rand_rois = bbox2roi([torch.cat([gt_bbox, random_bbox], dim=0) for gt_bbox, random_bbox in zip(gt_bboxes, rand_bboxes)])
        
        bbox_results = self._bbox_forward(x, rois, distilled_feat, gt_rand_rois, gt_labels, bg_feats)

        
        if self.use_bg_pro_as_ns:
            bbox_targets_ori = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, concat=False)
            
            labels, label_weights, bbox_targets, bbox_weights = bbox_targets_ori
            # concat the labels, label_weights, bbox_targets, bbox_weights
            # the labels should be bg label, label_weights should be the same as
            # other label. bbox_weights should be zero
            bg_labels = [torch.full((bg_bboxes[i].shape[0], ),
                                     self.bbox_head.num_classes,
                                     dtype=torch.long).cuda() for i in range(len(bg_bboxes))]
            bg_label_weights = [torch.full((bg_bboxes[i].shape[0], ),
                                     1.0,
                                     dtype=torch.long).cuda() for i in range(len(bg_bboxes))]
            bg_bbox_targets = [torch.zeros(bg_bboxes[i].shape[0], 4).cuda() for i in range(len(bg_bboxes))]
            bg_bbox_weights = [torch.zeros(bg_bboxes[i].shape[0], 4).cuda() for i in range(len(bg_bboxes))]
            # concat inside first
            labels = [torch.cat([label, bg_label], dim=0).cuda() for label, bg_label in zip(labels, bg_labels)]
            label_weights = [torch.cat([label_weight, bg_label_weight], dim=0).cuda() for label_weight, bg_label_weight in zip(label_weights, bg_label_weights)]
            bbox_targets = [torch.cat([bbox_target, bg_bbox_target], dim=0).cuda() for bbox_target, bg_bbox_target in zip(bbox_targets, bg_bbox_targets)]
            bbox_weights = [torch.cat([bbox_weight, bg_bbox_weight], dim=0).cuda() for bbox_weight, bg_bbox_weight in zip(bbox_weights, bg_bbox_weights)]
            # concat outside
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            bbox_targets = (labels, label_weights, bbox_targets, bbox_weights)
        else:
            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
            
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

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

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
