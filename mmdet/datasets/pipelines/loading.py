# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import final
from unittest import result

from isort import file
from pyparsing import rest_of_line
import torch

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import json

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 channel_order=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.channel_order=channel_order

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        if self.channel_order == None:
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        else:
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations,
              self).__init__(with_bbox, with_label, with_mask, with_seg, True,
                             file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])

        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
    """

    def __init__(self, min_gt_bbox_wh):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            return None
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results


@PIPELINES.register_module()
class LoadCLIPFeat:
    """Load pred-trained feat.
    """

    def __init__(self,
                 file_path_prefix=None,
                 num_of_rand_bbox=20,
                 select_fixed_subset=None):
        self.file_path_prefix = file_path_prefix
        # the path should like this
        # /data/zhuoming/detection/coco/feat
        self.gt_feat_prefix = osp.join(self.file_path_prefix, 'gt')
        self.random_feat_prefix = osp.join(self.file_path_prefix, 'random')
        self.num_of_rand_bbox = num_of_rand_bbox
        self.select_fixed_subset = select_fixed_subset

    def __call__(self, results):
        '''load the pre-extracted CLIP feat'''
        file_name = '.'.join(results['img_info']['filename'].split('.')[:-1]) + '.json'

        # load the gt feat
        gt_file_name = osp.join(self.gt_feat_prefix, file_name)
        gt_file_content = json.load(open(gt_file_name))
        
        #'feat', 'bbox', 'gt_labels', 'img_metas'
        gt_feat = np.array(gt_file_content['feat']).astype(np.float32)
        gt_bbox = np.array(gt_file_content['bbox']).astype(np.float32)
        gt_labels = gt_file_content['gt_labels']
        img_metas = gt_file_content['img_metas']
        
        ## filter the feat for the specifc category
        # compare the scale factor the pre-extract and the now pipeline
        pre_extract_scale_factor = np.array(img_metas['scale_factor']).astype(np.float32)
        now_scale_factor = results['scale_factor']
        now_gt_bbox = results['gt_bboxes']
        pre_extract_gt_bbox = gt_bbox
        
        # in here we round the scale factor in 6 decimals
        if (np.round(pre_extract_scale_factor, 6) == np.round(now_scale_factor, 6)).all():
            #print('in the matching')
            original_pre_extract_gt_bbox = pre_extract_gt_bbox
            original_now_gt_bbox = now_gt_bbox
        else:
            original_pre_extract_gt_bbox = pre_extract_gt_bbox / pre_extract_scale_factor
            original_now_gt_bbox = now_gt_bbox / now_scale_factor

        match_idx = []
        for now_bbox in original_now_gt_bbox:
            now_match_id = None
            match_distance = 10000000
            for i, extract_bbox in enumerate(original_pre_extract_gt_bbox):
                #print(extract_bbox, now_bbox)
                #print(np.round(extract_bbox, 2), np.round(now_bbox, 2))
                #print('extract_bbox, now_bbox', (np.round(extract_bbox, 2) == np.round(now_bbox, 2)))
                #if (np.round(extract_bbox, 0) == np.round(now_bbox, 0)).all():
                now_dist = (np.abs(extract_bbox - now_bbox)).sum()
                if now_dist < match_distance:
                    match_distance = now_dist
                    now_match_id = i
            match_idx.append(now_match_id)
            #print(now_bbox, original_pre_extract_gt_bbox[now_match_id])
            #if gt_bbox_matched == False:
            #    print('gt bbox is not matched',now_bbox)
                    
        # filter and reorder the feat
        match_idx = np.array(match_idx).astype(np.int32)
        #print('match_idx', match_idx, type(match_idx))
        remaining_feat = torch.from_numpy(gt_feat[match_idx])

        # pad the result
        if len(remaining_feat) < 100:
            padded_len = 100 - len(remaining_feat)
            padded_results = torch.zeros([padded_len] + list(remaining_feat.shape[1:]))
            remaining_feat = torch.cat([remaining_feat, padded_results], dim=0)        
        
        results['gt_feats'] = remaining_feat
        
        # load the random feat
        rand_file_name = osp.join(self.random_feat_prefix, file_name)
        rand_file_content = json.load(open(rand_file_name))
        
        # obtain the random bbox
        rand_feat = np.array(rand_file_content['feat']).astype(np.float32)
        rand_bbox = np.array(rand_file_content['bbox']).astype(np.float32)
        # selecting the subset of the file
        if self.select_fixed_subset != None:
            rand_feat = rand_feat[:self.select_fixed_subset]
            rand_bbox = rand_bbox[:self.select_fixed_subset]
        if rand_bbox.shape[-1] == 5:
            rand_bbox = rand_bbox[:, :4]
        rand_img_metas = rand_file_content['img_metas']
        
        # compare the scale factor and reshape the random bbox
        if (np.round(pre_extract_scale_factor, 6) == np.round(now_scale_factor, 6)).all():
            final_rand_bbox = rand_bbox
        else:
            final_rand_bbox = rand_bbox / pre_extract_scale_factor
            final_rand_bbox = final_rand_bbox * now_scale_factor
        
        # filter the random bbox we need
        random_choice = np.random.choice(rand_feat.shape[0], self.num_of_rand_bbox, replace=False)
        final_rand_bbox = final_rand_bbox[random_choice]
        final_rand_feat = rand_feat[random_choice]
        results['rand_bboxes'] = torch.from_numpy(final_rand_bbox)
        results['rand_feats'] = torch.from_numpy(final_rand_feat)
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str