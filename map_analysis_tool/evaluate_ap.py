from matplotlib.pyplot import annotate
import mmcv
import warnings
import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval
import numpy as np
from collections import OrderedDict


class COCO(_COCO):
    def __init__(self, annotation_file=None):
        if getattr(pycocotools, '__version__', '0') >= '12.0.2':
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
                UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs
    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)
    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)
    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)
    def load_anns(self, ids):
        return self.loadAnns(ids)
    def load_cats(self, ids):
        return self.loadCats(ids)
    def load_imgs(self, ids):
        return self.loadImgs(ids)

# just for the ease of import
COCOeval = _COCOeval

CLASSES_65 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
 'train', 'truck', 'boat', 'bench', 'bird', 'cat', 'dog', 
 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
 'skis', 'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 
 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
 'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 
 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 
 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'toothbrush')


CLASSES_48 = ('person', 'bicycle', 'car', 'motorcycle', 'train', 
            'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 
            'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 
            'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 
            'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 
            'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'microwave', 'oven', 'toaster', 
            'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

CLASSES_17 = ('airplane', 'bus', 'cat', 'dog', 'cow', 
        'elephant', 'umbrella', 'tie', 'snowboard', 
        'skateboard', 'cup', 'knife', 'cake', 'couch', 
        'keyboard', 'sink', 'scissors')


# file_paths = ['/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/base_results.bbox.json',
#               '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48/base_results.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/base_results.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_results.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/base_results_e18.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro_with_filp/base_results.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_with_filp/base_results.bbox.json',
#               '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_with_filp/base_results_e18.bbox.json']

file_paths = [#'/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/novel_results_trick.bbox.json',
              #'/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/novel_results_trick.bbox.json',
              #'/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_repro/novel_results_trick_e18.bbox.json',
              #'/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_200clip_pro_with_filp/novel_results_trick.bbox.json',
              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_with_filp/novel_results_trick.bbox.json',
              '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro_with_filp/novel_results_trick_e18.bbox.json']

#ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'
ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'
#ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_base48.json'

coco_object = COCO(ann_file)
# preparing the hyper-parameter of the evaluation
#cat_ids = coco_object.get_cat_ids(cat_names=CLASSES_65)
#cat_ids = coco_object.get_cat_ids(cat_names=CLASSES_48)
cat_ids = coco_object.get_cat_ids(cat_names=CLASSES_17)
img_ids = coco_object.get_img_ids()
proposal_nums=(100, 300, 1000)
#iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
iou_thrs = np.array([0.5])
metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
cocoGt = coco_object

for file_path in file_paths:
    predictions = mmcv.load(file_path)
    cocoDt = cocoGt.loadRes(predictions)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.catIds = cat_ids
    cocoEval.params.imgIds = img_ids
    cocoEval.params.maxDets = list(proposal_nums)
    cocoEval.params.iouThrs = iou_thrs
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()