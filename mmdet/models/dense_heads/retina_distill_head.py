# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
import torch
from mmdet.models.utils import build_linear_layer

@HEADS.register_module()
class RetinaDistillHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 clip_dim=512,
                 
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='map_to_clip',
                         std=0.01,
                         bias_prob=0.01)),
                 cls_predictor_cfg=dict(type='Linear'),
                 fg_vec_cfg=dict(load_path='data/embeddings/base_finetuned_48cates.pt'),
                 distill_loss_factor=1.0,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.clip_dim = clip_dim
        self.cls_predictor_cfg = cls_predictor_cfg
        self.fg_vec_cfg = fg_vec_cfg
        self.distill_loss_factor = distill_loss_factor
        super(RetinaDistillHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.distillation_loss_config = dict(type='L1Loss', loss_weight=1.0)
        self.distillation_loss = build_loss(self.distillation_loss_config)

    def init_weights(self):
        """Init module weights."""
        # Training Centripetal Model needs to reset parameters for Conv2d
        super(RetinaDistillHead, self).init_weights()
        
        # load the module and set the require_grad
        with torch.no_grad():
            self.retina_cls.weight.copy_(self.load_value)
        for param in self.retina_cls.parameters():
            param.requires_grad = False
        self.load_value.require_grad = False

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # from hidden dimension(feat_channels) converted to the anchor * clip_dim
        self.map_to_clip = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.clip_dim,
            3,
            padding=1)   
        # self.retina_cls = nn.Conv2d(
        #     self.feat_channels,
        #     self.num_anchors * self.cls_out_channels,
        #     3,
        #     padding=1)
        # using linear layer for converting anchor * clip_dim to anchor * num_of_classes
        self.retina_cls = build_linear_layer(self.cls_predictor_cfg,
                                in_features=self.clip_dim,
                                out_features=self.num_classes,
                                bias=False)
        
        load_value = torch.load(self.fg_vec_cfg.load_path)
        load_value = load_value.cuda()
        load_value = load_value / load_value.norm(dim=-1, keepdim=True)
        #load_value = load_value.t()
        self.load_value = load_value
        # use the matrix multiplication to finish the classification
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_feat = self.map_to_clip(cls_feat)
        # convert the dimension that fit the linear layer
        # the feature dim is torch.Size([2, 4608, 100, 152]), [bs, self.num_anchors * self.clip_dim, h, w]
        # permute the dimension
        cls_feat = cls_feat.permute([0,2,3,1])
        cls_feat = cls_feat.reshape(list(cls_feat.shape)[:-1] + [self.num_anchors, self.clip_dim])
        #feature after the conversion: torch.Size([2, 160, 100, 9, 512])
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
        
        # use the linear layer for classification
        cls_score = self.retina_cls(cls_feat)
        #cls_score after the linear classification: torch.Size([2, 160, 100, 9, 48])
        # map the classification score back to the dimension needed by the retina        
        cls_score = cls_score.reshape(list(cls_score.shape)[:-2] + [-1, ])
        #cls_score after the conversion: torch.Size([2, 160, 100, 432])
        # permute the dimension back to the suitable dimension
        cls_score = cls_score.permute([0,3,1,2])
        # cls_score after the permute: torch.Size([2, 432, 160, 100])
        
        bbox_pred = self.retina_reg(reg_feat)
        if self.training:
            return cls_score, bbox_pred, cls_feat
        else:
            return cls_score, bbox_pred

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_feat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             gt_feats=None, 
             rand_bboxes=None, 
             rand_feats=None,
             rand_bbox_weights=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        num_imgs = len(img_metas)
        
        # get the anchor of each grid
        # anchor_list 2 5 [torch.Size([136800, 4]), torch.Size([34200, 4]), torch.Size([8550, 4]), torch.Size([2223, 4]), torch.Size([630, 4])]
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        #print('anchor_list', len(anchor_list), len(anchor_list[0]), [ele.shape for ele in anchor_list[0]], [ele.shape for ele in anchor_list[1]])
        
        # match the gt bbox with each anchor
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        ###############################distillation######################################
        #all distillation bbox and all distillation feature
        all_dist_bboxes = [torch.cat([gt_bboxes_per_img, rand_bboxes_per_img], dim=0) for gt_bboxes_per_img, rand_bboxes_per_img in zip(gt_bboxes, rand_bboxes)] 
        all_target_feat = [torch.cat([gt_feats_per_img, rand_feats_per_img], dim=0) for gt_feats_per_img, rand_feats_per_img in zip(gt_feats, rand_feats)]
        #print('all_dist_bboxes', [ele.shape for ele in all_dist_bboxes])
        #print('all_target_feat', [ele.shape for ele in all_target_feat])
        
        #assign the distillation bbox with the anchor
        # we need the idx in the distillation feature that assigned to each anchor
        # pos_inds_list represent the idx of the anchor which is regared as the positive sample
        # the all_assigned_idx represent all the gt feature idx for each anchor, if the anchor label assigned is the -1 mean this anchor is a nagative anchor. 
        # which do no have high overlap with any distillation bbox
        pos_inds_list, all_assigned_idx = self.get_targets_idx(
            anchor_list,
            valid_flag_list,
            all_dist_bboxes,
            img_metas,
            gt_bboxes_ignore_list=None,
            gt_labels_list=None,
            label_channels=None)

        # select the anchor assigned idx
        pos_anchor_assigned_idx = [all_gt_idx[pos_idx] for pos_idx, all_gt_idx in zip(pos_inds_list, all_assigned_idx)]
        #print('pos_anchor_assigned_idx', pos_anchor_assigned_idx, [(-1 in ele) for ele in pos_anchor_assigned_idx])
        #print('pos_inds_list', [ele.shape for ele in pos_inds_list])
        
        # prepare the predicted feature
        #print('cls_feat', [ele.shape for ele in cls_feat])
        #cls_feat [torch.Size([2, 160, 100, 9, 512]), torch.Size([2, 80, 50, 9, 512]), torch.Size([2, 40, 25, 9, 512]), torch.Size([2, 20, 13, 9, 512]), torch.Size([2, 10, 7, 9, 512])]
        #converted class feature [torch.Size([191970, 512]), torch.Size([191970, 512])]
        concat_feat_list = []
        for i in range(num_imgs):
            all_feat_per_img = []
            for lvl in range(len(cls_feat)):
                all_feat_per_img.append(cls_feat[lvl][i].reshape(-1, self.clip_dim))
            concat_feat_list.append(torch.cat(all_feat_per_img, dim=0))
        
        all_predicted_feat = [feat[valid_idx, :] for feat, valid_idx in zip(concat_feat_list, pos_inds_list)]
        cat_all_predicted_feat = torch.cat(all_predicted_feat, dim=0)    
        #print('all_predicted_feat', [ele.shape for ele in all_predicted_feat])
        #print('converted class feature', [ele.shape for ele in concat_feat_list])
        #print('all_assigned_idx', [ele.shape for ele in all_assigned_idx])
        
        
        # prepare the target gt feature
        target_gt_feat = [gt_feat_set_per_img[gt_feat_idx_per_img] for gt_feat_set_per_img, gt_feat_idx_per_img in zip(all_target_feat, pos_anchor_assigned_idx)]
        cat_target_gt_feat =torch.cat(target_gt_feat, dim=0)
        #print('target_gt_feat', [ele.shape for ele in target_gt_feat])
        
        # prepare the distillation weight
        # if rand_bbox_weights:
        #     selected_distill_weight = [gt_feat_weight_per_img[gt_feat_idx_per_img] for gt_feat_weight_per_img, gt_feat_idx_per_img in zip(rand_bbox_weights, pos_anchor_assigned_idx)]
        #     selected_distill_weight = torch.cat(selected_distill_weight, dim=0)
        
        # calculate the distllation loss
        #cat_all_predicted_feat = cat_all_predicted_feat / cat_all_predicted_feat.norm(dim=-1, keepdim=True)
        distill_loss_value = self.distillation_loss(cat_target_gt_feat, cat_all_predicted_feat)
        distill_loss_value *= (self.clip_dim * self.distill_loss_factor)
        
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, distill_loss=distill_loss_value)

    ### update the input and the input for each function
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      gt_feats=None,
                      rand_bboxes=None,
                      rand_feats=None,
                      rand_bbox_weights=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore,
                            gt_feats=gt_feats, 
                            rand_bboxes=rand_bboxes, 
                            rand_feats=rand_feats,
                            rand_bbox_weights=rand_bbox_weights)
        
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def get_targets_idx(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        #if gt_labels_list is None:
        # use the gt_label to represent which distillation bbox be assigned
        gt_labels_list = [torch.tensor([k for k in range(len(gt_bboxes_list_per_image))]).cuda() for gt_bboxes_list_per_image in gt_bboxes_list] 
        #print('gt_labels_list', gt_labels_list)
        label_channels = max([len(ele) for ele in gt_bboxes_list])
        results = multi_apply(
            self._get_targets_single_idx,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        
        #print('pos_inds_list', len(pos_inds_list), [ele.shape for ele in pos_inds_list], pos_inds_list[0])
        #print('all_labels', len(all_labels), [ele.shape for ele in all_labels], all_labels[0], [(ele!=-1).sum() for ele in all_labels])
        # if the anchor idx is -1 means the anchor is not assigned
        # pos_inds_list 2 [torch.Size([2713]), torch.Size([3080])] tensor([ 33701,  33710,  33719,  ..., 191950, 191952, 191958], device='cuda:1')
        # all_labels 2 [torch.Size([191970]), torch.Size([148851])] tensor([-1, -1, -1,  ..., -1, -1, -1], device='cuda:1') [tensor(2713, device='cuda:1'), 
        # tensor(3080, device='cuda:1')]
        
        # pos_inds_list represent the idx of the anchor which is regared as the positive sample
        # the all_labels represetn all the gt feature idx for each anchor, if the anchor label assigned is the -1 mean this anchor is a nagative anchor. 
        # which do no have high overlap with any distillation bbox
        return pos_inds_list, all_labels
        
        # # no valid anchors
        # if any([labels is None for labels in all_labels]):
        #     return None
        # # sampled anchors of all images
        # num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        # num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # # split targets to a list w.r.t. multiple levels
        # labels_list = images_to_levels(all_labels, num_level_anchors)
        # label_weights_list = images_to_levels(all_label_weights,
        #                                       num_level_anchors)
        # bbox_targets_list = images_to_levels(all_bbox_targets,
        #                                      num_level_anchors)
        # bbox_weights_list = images_to_levels(all_bbox_weights,
        #                                      num_level_anchors)
        # res = (labels_list, label_weights_list, bbox_targets_list,
        #        bbox_weights_list, num_total_pos, num_total_neg)
        # if return_sampling_results:
        #     res = res + (sampling_results_list, )
        # for i, r in enumerate(rest_results):  # user-added return values
        #     rest_results[i] = images_to_levels(r, num_level_anchors)

        # return res + tuple(rest_results)

    def _get_targets_single_idx(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  -1,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=-1)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
