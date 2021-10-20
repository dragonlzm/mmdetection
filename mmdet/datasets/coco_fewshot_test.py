# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

from numpy.lib.function_base import average

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
from .coco import CocoDataset
import torch


@DATASETS.register_module()
class CocoFewshotTestDataset(CocoDataset):

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        #results = dict(img_info=img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        # the results is a list, in which each ele is a tensor(0., device='cuda:0')
        #print(results)
        #result = torch.mean(torch.cat([ele['acc'].cpu().unsqueeze(dim=0).float() for ele in results]))
        result = torch.sum(torch.cat([ele[0].cpu().unsqueeze(dim=0).float() * ele[1] for ele in results]))
        all_bbox_num = sum([ele[1] for ele in results])
        result /= all_bbox_num

        # for confusion matrix
        confu_mat_result = torch.zeros(21,21)
        for res in results:
            pred_list = res[2]
            gt_list = res[3]
            for pred_label, gt_label in zip(pred_list, gt_list):
                confu_mat_result[gt_label][pred_label] += 1
        print(confu_mat_result)

        # for logit aggregation
        '''
        top2_case_aggre_per_class = torch.zeros(21, 1000)
        for img_result in results:
            gt_list = img_result[3]
            topk_resilt = img_result[4]
            # the topk_resilt is a tuple, in which the first ele is top3 value,
            # the second ele is the top3 index
            # in here we first aggregate for each class which logit
            # is the most commonly appear logit
            # the each row of the top2_case_aggre_per_class shows that the times 
            # for each ele in the 1000 dim result become the second highest logit.
            for i, gt_label in enumerate(gt_list):
                for j in range(3):
                    top2_idx = topk_resilt[i][j]
                    top2_case_aggre_per_class[gt_label][top2_idx] += 1

        # for each class we find the idx of the logit which appear most as the second highest logit
        print(torch.topk(top2_case_aggre_per_class, 10))'''


        eval_results = {}
        eval_results['acc'] = result.item()
        return eval_results
