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

#file_path = '/data/zhuoming/detection/uk_rpn/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48_no_sigmoid/novel_results.bbox.json'
#file_path = '/data/zhuoming/detection/uk_rpn/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_trirpn_1x_coco_base48_no_sigmoid/base_results.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/base_results_65cates.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/novel_results_65cates.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/novel_gt_result.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/fixed_label_prediction.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/filter_fp_prediction.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_gn_10_200clipproposal/add_fn_prediction.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_use_bg_pro_60/novel_results_65cates.bbox.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results.bbox.json'
#file_path = '/home/zhuoming/final_pred_v1.json'
#file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_2x_coco_base48_200clip_pro/base_results_e18.bbox.json'

#file_path = '/data/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/filter_iop_prediction.json'
file_path = '/data/zhuoming/detection/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/filter_iop_prediction.json'

#ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_65cates.json'
#ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_novel17.json'
ann_file = '/data/zhuoming/detection/coco/annotations/instances_val2017_base48.json'

coco_object = COCO(ann_file)

# preparing the hyper-parameter of the evaluation
#cat_ids = coco_object.get_cat_ids(cat_names=CLASSES_65)
cat_ids = coco_object.get_cat_ids(cat_names=CLASSES_48)
img_ids = coco_object.get_img_ids()
proposal_nums=(100, 300, 1000)
iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
#iou_thrs = np.array([0.5])
metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']


cocoGt = coco_object
predictions = mmcv.load(file_path)
cocoDt = cocoGt.loadRes(predictions)


cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.catIds = cat_ids
cocoEval.params.imgIds = img_ids
cocoEval.params.maxDets = list(proposal_nums)
cocoEval.params.iouThrs = iou_thrs

coco_metric_names = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
    'AR@100': 6,
    'AR@300': 7,
    'AR@1000': 8,
    'AR_s@1000': 9,
    'AR_m@1000': 10,
    'AR_l@1000': 11
}

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

eval_results = OrderedDict()
for metric_item in metric_items:
    key = f'bboxes_{metric_item}'
    val = float(
        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
    )
    eval_results[key] = val

print(cocoEval.eval['precision'].shape)
#(10, 101, 65, 4, 3)
print(cocoEval.params.recThrs)
# [0.   0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1  0.11 0.12 0.13
#  0.14 0.15 0.16 0.17 0.18 0.19 0.2  0.21 0.22 0.23 0.24 0.25 0.26 0.27
#  0.28 0.29 0.3  0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4  0.41
#  0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5  0.51 0.52 0.53 0.54 0.55
#  0.56 0.57 0.58 0.59 0.6  0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69
#  0.7  0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8  0.81 0.82 0.83
#  0.84 0.85 0.86 0.87 0.88 0.89 0.9  0.91 0.92 0.93 0.94 0.95 0.96 0.97
#  0.98 0.99 1.  ]
all_precision = cocoEval.eval['precision']
with open('test.npy', 'wb') as f:
    np.save(f, all_precision)

import torch
all_precision = torch.from_numpy(all_precision)
all_precision = all_precision.permute([1,0,2,3,4])
precision_over_recall = [torch.mean(ele[ele>-1]).item() for ele in all_precision]
print(len(precision_over_recall), sum(precision_over_recall)/len(precision_over_recall))
print(precision_over_recall)
# precision_over_recall = torch.mean(all_precision[all_precision>-1], dim=(1,2,3,4))
#print(np.mean(temp[temp>-1]))
#0.3284804737520371


# visulization
import matplotlib as mpl
import matplotlib.pyplot as plt
x = [0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13
,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27
,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41
,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55
,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69
,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83
,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97
,0.98,0.99,1.,]
# base_y=[0.8293181887133676, 0.7931063172741163, 0.7633451337242759, 
#         0.7427780956870432, 0.7262229718615569, 0.7122512187103852, 
#         0.6917808956877, 0.6808351299514392, 0.6679963953984451, 
#         0.6570176471745749, 0.6476103979909864, 0.6354206562857616, 
#         0.6250861646360141, 0.6156410739573375, 0.6075577501628038, 
#         0.5913820048204824, 0.5817274061734342, 0.5728288272052573, 
#         0.5642029726707156, 0.557516206973082, 0.5502973462897156, 
#         0.536508369376122, 0.527086817258789, 0.517483400740076, 
#         0.5102627423702993, 0.5040323715824889, 0.4933612009352585, 
#         0.4859249349680214, 0.4771868229414248, 0.46880052508475767, 
#         0.4623128020376746, 0.45437302732764223, 0.44669028998025845, 
#         0.43923250474517367, 0.42727326818971584, 0.4204006285879569, 
#         0.4127120024801429, 0.40514053547437295, 0.398709327078212, 
#         0.3907913605959056, 0.38484534122981146, 0.3771390073019012, 
#         0.3697601184036317, 0.36258323550131705, 0.3558900573181154, 
#         0.349300263991941, 0.34222963572077725, 0.33411506665066465, 
#         0.32741811712491603, 0.3210524399320401, 0.3162422340907629, 
#         0.2994493381936549, 0.2934336190729968, 0.28649122261997473, 
#         0.2799402245140299, 0.27348270163495586, 0.26641184285642877, 
#         0.25856624596696287, 0.25120113211342054, 0.2440790854756159, 
#         0.239011514814986, 0.23064391472864937, 0.22325847769310603, 
#         0.2170487197394041, 0.21012655813568376, 0.2030457619710547, 
#         0.19515370825636924, 0.18595013667941146, 0.1794890012625989, 
#         0.17251197060556026, 0.1651884688705093, 0.1592952561991061, 
#         0.1525498066828376, 0.14424366833015256, 0.13807372121899936, 
#         0.1323645208048313, 0.12525390714322657, 0.1177758339596525, 
#         0.10861159502504367, 0.10252267072033355, 0.09672137929210509, 
#         0.0901489332853848, 0.08345604036642972, 0.07767524178278871, 
#         0.06923408820592455, 0.06381949202746136, 0.05680502259000153, 
#         0.05159423888885363, 0.044680314408862346, 0.03964005473562236, 
#         0.03600779572653488, 0.03129089648939861, 0.027810630974274553, 
#         0.024126368784161134, 0.020113557811023794, 0.017820298455737044, 
#         0.014868041985221585, 0.012850824268004019, 0.010934374765425395, 
#         0.009797671645157262, 0.009176706809144378]
# novel_y = [0.8468289391712415, 0.8192170187908953, 0.7925871647191154, 
#            0.7750011529736598, 0.7589447159795967, 0.7473283212941294, 
#            0.7265891849596255, 0.717120872624342, 0.7024693467592179, 
#            0.6902793229909789, 0.68117946767374, 0.6681725444376772, 
#            0.6568118529765622, 0.643924241301934, 0.6372812229730697, 
#            0.6260383289786776, 0.6167146728900571, 0.6079110887127359, 
#            0.5998311126160333, 0.5927123362150951, 0.5863743710193872, 
#            0.5753557066615878, 0.5675347394213481, 0.55769423261846, 
#            0.5491871934007991, 0.5428858568271796, 0.5299454432497493, 
#            0.5238040791387941, 0.514785523942957, 0.5067767093867802, 
#            0.5000094769411275, 0.49133516013890505, 0.48456051078555595, 
#            0.47792759281935415, 0.46899448648220826, 0.46166142591017023, 
#            0.453484457299936, 0.4461428853004338, 0.43866124202301704, 
#            0.43069614778637183, 0.4246184986999233, 0.41464783168556013, 
#            0.4079494949460846, 0.39947737026452246, 0.3928562505786106, 
#            0.3856324944916786, 0.379324069288054, 0.37036943555827406, 
#            0.36336751605071793, 0.35734733583146766, 0.3516368985436595, 
#            0.33262976057642485, 0.32576926636884224, 0.3182574742474945,
#            0.311181547811726, 0.3040781259602019, 0.2958871117150965, 
#            0.2883216532573259, 0.281890313228015, 0.2741954950263533, 
#            0.2681904334706938, 0.25695063943324437, 0.24871803911346646, 
#            0.24122076108907747, 0.23431323519787933, 0.22744211277239598, 
#            0.21961631431133438, 0.20817272801370812, 0.2011224031585182, 
#            0.19351261863151348, 0.18548507583631585, 0.17881221602817104, 
#            0.1714291132064254, 0.16247700647962066, 0.15590701080650268, 
#            0.149137458548474, 0.14144553363473444, 0.1341908691957253, 
#            0.12572735888756784, 0.11810795741082178, 0.11202898118221756, 
#            0.10369672556782651, 0.09683599265041308, 0.09097614851401216, 
#            0.08335713133684533, 0.07726109103810032, 0.07080203509426765, 
#            0.06408455683148806, 0.05853780659920449, 0.051517076913084915, 
#            0.046619813480559734, 0.04037233070670526, 0.035417258453984975, 
#            0.029764055528755013, 0.02505589270703687, 0.020354233413252647, 
#            0.016436943358263463, 0.01313828222411547, 0.010586286562286064, 
#            0.00828260577621895, 0.007613479946836482]
# plt.plot(x, base_y , color='red', label='mAP = 0.328')
# plt.plot(x, novel_y , color='green', label='mAP = 0.355')

# base_y = [0.8447217621071962, 0.807090629390678, 0.7777217410515146, 0.7550662918200793, 0.7384290791125824, 0.7224424727101451, 0.7048670101030364, 0.6924654447511227, 0.6775388291013794, 0.6661802404062163, 0.6573203806406231, 0.646228388072737, 0.6347533072710219, 0.6242604621242516, 0.6173548918315597, 0.602630137457733, 0.5932388654563197, 0.5829833399154918, 0.5751559505485719, 0.5675724641077505, 0.5605122671655118, 0.5498329571099649, 0.5429645122415866, 0.5350684636244486, 0.5275962838246345, 0.5222082729983955, 0.5112689898194189, 0.5043764565230132, 0.4965560619322763, 0.4882632900535897, 0.48210867660410445, 0.4739182369867076, 0.466535538265229, 0.4593104005678428, 0.450960048335987, 0.4427978014787876, 0.4347388669201482, 0.42784946040508753, 0.42157503424157833, 0.4129832115733218, 0.4070332810046112, 0.3987124442022736, 0.39200340095713926, 0.3844143975038685, 0.3774474066236692, 0.3693983478181009, 0.36205311678121943, 0.3554819765326312, 0.3491005095802719, 0.34249172342811063, 0.3369567108852815, 0.3176632596875053, 0.31111801298071773, 0.3038029998613979, 0.29561402836664974, 0.28940477331317527, 0.281644563249878, 0.2750628129779969, 0.26929730506931776, 0.26242334886225627, 0.2571083145310356, 0.24801326399705398, 0.23991400295029833, 0.2327307898098159, 0.2243881082117189, 0.2150949400658262, 0.2079087657937204, 0.19735588072905932, 0.19055195860400517, 0.18366644566235632, 0.17564705671881803, 0.1689594780724355, 0.16273118951071247, 0.15444443015673898, 0.14774727257431208, 0.1412290105370252, 0.13344730489662557, 0.12553719675498487, 0.11618415173384176, 0.10987485343120977, 0.1046359699168071, 0.09752209380867703, 0.0905965739258574, 0.08434168071469914, 0.07678535087439038, 0.07027466291233618, 0.06444124808522722, 0.058215741842537444, 0.05137598659957402, 0.044561829245019266, 0.03966311950428144, 0.034332025278761086, 0.029963103227322485, 0.025784472800171473, 0.021451212887006577, 0.01827332051692813, 0.014359019250735228, 0.01187677121104506, 0.009916431011852384, 0.008513389463732163, 0.008108659678193129]
# novel_y = [0.5983272442885412, 0.5649283471111336, 0.5231127816848139, 0.4897888148608, 0.47085128041688334, 0.45641461106647413, 0.4314389132599257, 0.41682320020293717, 0.40033339421525926, 0.38267926877675135, 0.37653011019552396, 0.3646160517128149, 0.3468670913828559, 0.3367106871333424, 0.3246617318789771, 0.3161010558955712, 0.3046766373044399, 0.2904454159110654, 0.2833876371554195, 0.2783571824515809, 0.272088266851994, 0.25988735777472416, 0.25097404039097426, 0.24481647760943223, 0.23865226128949482, 0.23155980507500998, 0.22228355273341344, 0.21384335661185053, 0.20711913491369938, 0.20249450383999665, 0.19687500093080107, 0.1892029524525358, 0.18555751639867468, 0.17849222964107303, 0.17229523058799526, 0.16795377767971695, 0.1637783094162995, 0.15970179964576758, 0.15391043551811828, 0.15052579565852034, 0.1461356081647149, 0.13854657826007333, 0.13445999079599671, 0.1307475247334012, 0.12665106968677434, 0.12243376628181439, 0.11890644935612016, 0.11521335143661056, 0.11096193268674638, 0.10618451762411303, 0.1043583284061621, 0.09848291154708869, 0.09444175705492076, 0.09130231361635822, 0.0877910101669339, 0.08472320772930639, 0.08180002756096891, 0.07805536597983835, 0.0752852243637957, 0.07168813517830354, 0.06886789865745636, 0.06565338227753705, 0.0626689379822846, 0.0596726463388187, 0.05736602294797664, 0.053953152173865634, 0.05150899293540402, 0.04916273207729971, 0.046152072838354964, 0.0437359831463212, 0.04141592987468555, 0.03896593667851146, 0.03716088077595757, 0.03482497748286407, 0.0325185550222492, 0.030117276577954915, 0.02752942475955588, 0.02550287333696153, 0.02381302039026227, 0.02172469945882878, 0.020325705469842515, 0.01869294629349571, 0.016892137746673083, 0.015401503836485734, 0.013804117929213431, 0.012181428687347406, 0.011157623620302756, 0.009676494415275806, 0.008155847453948255, 0.007259910894168613, 0.005766153680402249, 0.004331822944500602, 0.003633086668514754, 0.003099534358172933, 0.0026399659456655166, 0.0020375693391979787, 0.0011571916401413255, 0.0007316607222156111, 0.0004141876565899028, 0.00018867029319269245, 0.00018867029319269245]
# plt.plot(x, base_y , color='red', label='base')
# plt.plot(x, novel_y , color='green', label='novel')
optimal_result_y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
base_y = [0.9643666952260703, 0.9575310885597746, 0.9498916581032487, 0.9445533607186657, 0.9376082361291821, 0.9299383770097067, 0.9191652158087967, 0.9152043184515433, 0.907156701155031, 0.9010638708606807, 0.8955071579122832, 0.8861190343133745, 0.8781467258566712, 0.8737132687906972, 0.8701076439489053, 0.8575587113506232, 0.8485349272501074, 0.8385515508929101, 0.831176601594505, 0.8236577321486372, 0.8168804998070954, 0.8076205582686398, 0.8016708034219353, 0.795206697023431, 0.7895573222561441, 0.7851717127015859, 0.7742051541530423, 0.7681984566443668, 0.7605650071769525, 0.7513853375356208, 0.7448484354141532, 0.7380843624136758, 0.7307424152500414, 0.7245587235330047, 0.7190442655704746, 0.7119228872126336, 0.7038501319553032, 0.696562070687508, 0.6901269194924932, 0.6798271981323373, 0.6732779596343939, 0.663844182574012, 0.6569493257293898, 0.6487694164682744, 0.6426195441528896, 0.6339332113270726, 0.625237538117751, 0.6188486120645452, 0.6113160147040276, 0.6048067201691156, 0.5968794785805143, 0.5701096569143095, 0.5623849977279778, 0.5536751608004877, 0.5440032995159206, 0.5349359546469253, 0.5241157925763118, 0.5160678623935602, 0.5083227090475513, 0.49935449836265966, 0.4933570830594161, 0.4821220750403771, 0.47068448975838095, 0.46153419037095644, 0.4519802814074936, 0.4405192703549345, 0.42920139812955493, 0.4146409313486727, 0.4052589444024851, 0.3960812292885317, 0.38648950660128667, 0.37590002235193043, 0.3666187577056625, 0.35451609758690467, 0.3444645733450419, 0.3343518958412586, 0.32170187192096666, 0.3065748675190749, 0.2947563851510393, 0.2831487464704905, 0.2738675986494623, 0.25828717504900656, 0.243952809109529, 0.23073071940811207, 0.21433395826971913, 0.20235144995390644, 0.18785471289615935, 0.17447249432706202, 0.16229674167383457, 0.14256717343728803, 0.1288836270909708, 0.11148756289011377, 0.0992640004474062, 0.08882013554360994, 0.07435961759571678, 0.06319113402263138, 0.05052437222981691, 0.039703920585503344, 0.02930461113396242, 0.02120186883306798, 0.018731649787862965]
novel_y = [0.7909457889812436, 0.7754127427710668, 0.7583901168583188, 0.719159846359752, 0.7038295569730784, 0.6958075481477737, 0.6657763877342634, 0.6483710434226154, 0.6214127485406493, 0.6041045978001851, 0.5949527295600026, 0.5898621042444104, 0.5765276849023504, 0.5635382901659588, 0.5462856445657313, 0.5385072393729269, 0.5206789586500158, 0.5015542426683236, 0.4966462099985395, 0.48981230896211875, 0.48322827044982375, 0.46920142572611073, 0.46191240612857426, 0.45489197097347506, 0.44645723614690247, 0.43597409681933413, 0.425143737911287, 0.4111334237486244, 0.40292641716884064, 0.39423228459255855, 0.3897044569053568, 0.3768106412554921, 0.3735379616117428, 0.3572923461816972, 0.3467312000646917, 0.3394480030330004, 0.3338806544433885, 0.3301051051363891, 0.32353319346394716, 0.3190569740071039, 0.3123747761434829, 0.2934664915812451, 0.2849879943002165, 0.28048315070341695, 0.27354946089068144, 0.2670310752221578, 0.2608833035274455, 0.2575450005805133, 0.2509858955891709, 0.24101929370458897, 0.23866295008102992, 0.22638124175456936, 0.21925484722463368, 0.21451035225154003, 0.20634682845336685, 0.1997825826549159, 0.19481618553794586, 0.18767944350151142, 0.18295485769212771, 0.17522926721820478, 0.17142939396834714, 0.16448513668673292, 0.15871901110431388, 0.15359003060039697, 0.14869936040192852, 0.14193428176530343, 0.13644445970670196, 0.1316630696318258, 0.1257469261271332, 0.11978580723418437, 0.11642668876760853, 0.11090320005080725, 0.10680598547700992, 0.10131634923337428, 0.09575850515582485, 0.08670376156929387, 0.0787125221025146, 0.07473662242319036, 0.06948868264502699, 0.06460288242013872, 0.06061749617350245, 0.05666314676272182, 0.05400717976434004, 0.04978744287943067, 0.04682838372658358, 0.04328631044544755, 0.03857111223274519, 0.03503775700862666, 0.029963218857621755, 0.026553864433122937, 0.022904995210771825, 0.018125957153653234, 0.015074992303396418, 0.013731127655984783, 0.01211651849210043, 0.008620554936246982, 0.006719459419065463, 0.00460201345139651, 0.0031993262587316483, 0.0009441526247595448, 0.0009441526247595448]
#fixed_novel_y = [0.9004572586817957, 0.6738958281077824, 0.5317639388595061, 0.439499711132719, 0.4151746116212544, 0.38464988205009903, 0.3343536310037676, 0.3168466624654555, 0.30074663350985015, 0.2969431425047356, 0.29570442626513965, 0.2873311361971254, 0.2801746917552011, 0.2719334426355253, 0.26303123749279034, 0.25964545607253453, 0.25372183434979945, 0.24786233998007623, 0.23913898244658038, 0.23881732499214534, 0.23737557814769067, 0.23182439861287668, 0.23101455609571092, 0.21504842418167067, 0.2132013467560952, 0.21306344564702695, 0.20217221872774763, 0.20126104580308926, 0.20022875803734386, 0.1998778095847965, 0.19820719487685953, 0.189927628784385, 0.18820197190479915, 0.18766490395895816, 0.18599187036578013, 0.18520582108337366, 0.18390838174645652, 0.18362903205785563, 0.18210789163738883, 0.17979166533404828, 0.17934205191542918, 0.17805039688805882, 0.1764095057295653, 0.1753778351018456, 0.16881805528488344, 0.1665552527607644, 0.16601824512537647, 0.1559079612508886, 0.15550321263360928, 0.15521774711620015, 0.15515628071715054, 0.15189160537156296, 0.1514675876136008, 0.150965942480921, 0.14597043227624945, 0.1459432545931184, 0.1459181502117432, 0.14581294417919616, 0.14559363763250244, 0.1393279210697383, 0.13661846433014024, 0.13647139798288951, 0.1333358763786143, 0.13318747430380642, 0.12886207610801698, 0.12833922708028983, 0.12794351846534674, 0.12294530404964071, 0.122264944261637, 0.12215153494866358, 0.1163430144006123, 0.11491596497899241, 0.11323684181366424, 0.10613090255160473, 0.1054789468525129, 0.1054140051547476, 0.09331382662436383, 0.09296064776172695, 0.08537865632859501, 0.08527919598811691, 0.08514406564702443, 0.08473518834491835, 0.08038808050168601, 0.07976333748620582, 0.075184806676308, 0.07203283354608045, 0.07134862249870502, 0.06924976233558663, 0.06901944211285958, 0.0669890036187855, 0.06241764851013631, 0.058885789401960424, 0.05546334374017198, 0.052467786614504866, 0.04344391940500727, 0.03897137852012375, 0.0319213495152219, 0.030540884784233842, 0.02517128767348656, 0.009638490909430288, 0.009638490909430288]
filter_fp_novel_y = [0.9908088235294118, 0.9385055911478469, 0.9022916340630774, 0.8842505549151872, 0.8777108681022471, 0.8716875970356094, 0.8578267765120332, 0.8358212867260078, 0.8158911871963477, 0.7826811533224499, 0.7768317168498502, 0.7752812897693067, 0.773818114786395, 0.771102213360455, 0.771102213360455, 0.7707251394539694, 0.7678433436477481, 0.7671169275264486, 0.7658212336636191, 0.7638407764561799, 0.7458959982172839, 0.7444727277712431, 0.7293249280006812, 0.7169692008987065, 0.714090141451541, 0.7137542109082817, 0.6969541820732182, 0.6969148557966873, 0.695889670947973, 0.6946385384112618, 0.6932027936934112, 0.6793297333771504, 0.6786511253256136, 0.677800827710659, 0.6776964098294245, 0.6521986466586509, 0.6517371942792326, 0.651684287034072, 0.6359437438276583, 0.6359306369806413, 0.6351313342784992, 0.6342728288500461, 0.6196068582850413, 0.6190658118323248, 0.6188299086475704, 0.6033367988353586, 0.590735809316978, 0.5894861657668906, 0.5747724665144378, 0.5734729150793247, 0.573118478885169, 0.5578413349781948, 0.5574658355616369, 0.5571635877802864, 0.5569822650881432, 0.5567888464828162, 0.5271862574400293, 0.5271223224312114, 0.5010750552517081, 0.5002320080581407, 0.49042205273829165, 0.4893153696656013, 0.44963560784887485, 0.4390483202111877, 0.4389618150208763, 0.43889896936979533, 0.4388197744772567, 0.42587507563963783, 0.4039656982141419, 0.38995469418377, 0.3897839913076869, 0.37413923097708607, 0.3736928560358152, 0.3625772149154385, 0.33967375563326224, 0.3146413610135093, 0.2959730613989044, 0.29582148045578877, 0.29563883477141295, 0.2735245441809021, 0.2609821792814232, 0.24280290173960828, 0.2427110431338178, 0.22167940780821374, 0.2214079991874686, 0.21148828866594144, 0.20169686491540567, 0.19024553607382924, 0.17171117127910335, 0.15899970834663413, 0.13456371153564956, 0.09527188028604044, 0.08247054310779973, 0.08241284906617043, 0.07473984376058936, 0.05657525144415837, 0.04954200336231694, 0.04240137873893113, 0.032071801630755126, 0.01197841641674735, 0.01197841641674735]
filter_fp_base_y = [0.9624388728555395, 0.9080926175267628, 0.8756098625727122, 0.8573008275031153, 0.8422193025410164, 0.8276124513722548, 0.819783532136078, 0.812134615178211, 0.7993752574924348, 0.791933012060118, 0.7872445366208358, 0.7843538898316111, 0.7800613989892748, 0.7782715356593628, 0.7772989339556484, 0.7760795558582828, 0.7747167362433794, 0.7733496991814736, 0.7715025212954784, 0.768576693781073, 0.7677610262176846, 0.7657519358666816, 0.7649227172127699, 0.7626647956878596, 0.7619346939516833, 0.7611939435071098, 0.7561361680759603, 0.7556293560042882, 0.7537567670017363, 0.7522103919459631, 0.7513776447561162, 0.7489064855931513, 0.7482041736937841, 0.7477084865205857, 0.7429188935323352, 0.7423222667387338, 0.7418245040676548, 0.7411328292350134, 0.7400143491909383, 0.7391183977281384, 0.7388765367986839, 0.7379725878107569, 0.7367548098296354, 0.7310232474821218, 0.7302744454130452, 0.729633105097664, 0.7291966096246921, 0.728784171676897, 0.7280123862577843, 0.7225883775927165, 0.7220989016656998, 0.7137821377696777, 0.7086520152497485, 0.7076645472624222, 0.7070548178415585, 0.7066952449239392, 0.7020052754781461, 0.7012125693634448, 0.7010105036594756, 0.7008031139712811, 0.687124397790028, 0.6867334221921619, 0.6831010142958767, 0.6828337888115755, 0.6781046933935532, 0.6687574335611298, 0.6630925022210722, 0.650686244584599, 0.6461884937160276, 0.6367419787311336, 0.6320844204904768, 0.610394811107994, 0.6011212510643876, 0.5925199824948755, 0.5874978178983663, 0.586977374762066, 0.5693871136167792, 0.5516210161105244, 0.5340794200279905, 0.5229782774289944, 0.5072659048343914, 0.4778922221449809, 0.4716417143819953, 0.4676835644804872, 0.4514578607297543, 0.4338528309815205, 0.4235052361005776, 0.4006601359523915, 0.3801383547263596, 0.3564548422196349, 0.3486226545815334, 0.3315467097432074, 0.30035913055117297, 0.27500533181843634, 0.2638413657229937, 0.22330506789084462, 0.18807684008129322, 0.15354875720906777, 0.10627003482644456, 0.0730822229370885, 0.06085582409246401]

add_fn_base_y = [0.9835792824074076, 0.9835792824074076, 0.9835792824074076, 0.9830584490740739, 0.9830584490740739, 0.9816455501379038, 0.979913797767219, 0.9792627561005524, 0.9778414690090352, 0.9767749582486205, 0.9766069474959325, 0.9757269559641042, 0.9747033077335163, 0.9736681105693354, 0.9721595323693764, 0.9686767386287521, 0.9660071522662932, 0.9639680181023034, 0.9625713782537362, 0.959625007911355, 0.9582031245775933, 0.9538774153783254, 0.9496342539552997, 0.9453219363337964, 0.9423604978608265, 0.9400636218561897, 0.9330106894700716, 0.9302483058836102, 0.9252560451662151, 0.9203307998265066, 0.9164217922455317, 0.9123421692649545, 0.9069994316161076, 0.9035268004804043, 0.8983844081278602, 0.8945487161547435, 0.8864373526723698, 0.8808854088877547, 0.8756882307045928, 0.8702117665162599, 0.8648388028864169, 0.854790367135977, 0.8488855822028074, 0.8418217856922356, 0.8365074041341389, 0.8297624302786537, 0.8229563941843738, 0.817776033597329, 0.8118004204850426, 0.8059561991086964, 0.8007117807983954, 0.777047868690746, 0.7723057494209079, 0.7641588325742459, 0.7568710997786922, 0.7490130361867166, 0.7366962251759936, 0.7278513272030916, 0.7190217332490939, 0.7098564509142319, 0.7013413277111284, 0.6927263430054804, 0.6815067664822991, 0.6733285898819137, 0.6644797108203455, 0.6514907831349674, 0.6401275705802836, 0.625986238767754, 0.6152947429308896, 0.6050919207308252, 0.5922253372253715, 0.5811583583767774, 0.5661656738869298, 0.5516492584096687, 0.5407243610091671, 0.5297791045245979, 0.5175018565239187, 0.5037928758024793, 0.4894963108094104, 0.4754189081309752, 0.46262200462866965, 0.4386132289423837, 0.4263141258160547, 0.4088322682661346, 0.39070861258954787, 0.3720410438726137, 0.35310360480737857, 0.3346295420610132, 0.31444170690128714, 0.2846623929185437, 0.2629112385080608, 0.23985979008281744, 0.2182904464676617, 0.19408699935870166, 0.1734927659413282, 0.15395800232249057, 0.12986437678529128, 0.11154094553435599, 0.08980432456515011, 0.066100206230769, 0.04927684339162264]
add_fn_novel_y = [1.0, 1.0, 0.9986631016042781, 0.9986631016042781, 0.9986631016042781, 0.9986631016042781, 0.9888591800356508, 0.9888591800356508, 0.9730903034242326, 0.9718648132281541, 0.9718648132281541, 0.9703265230869228, 0.9653264576974415, 0.9611826438393588, 0.9558594711980608, 0.9553294864768997, 0.9429494720617927, 0.936649432012461, 0.9337458633648541, 0.9320595068179236, 0.9195987823486982, 0.9104635506735487, 0.9065377531694746, 0.893705325714704, 0.878669125957033, 0.8747053128700887, 0.8489895774180115, 0.8348123469559617, 0.8289024719025567, 0.8127934657700854, 0.7971270566004368, 0.7897143794084569, 0.7848379268516098, 0.7796150094843639, 0.7675996041042651, 0.7606374586504844, 0.7473882375989296, 0.7399149468594344, 0.7275370408135861, 0.7216041652499982, 0.7138751779843078, 0.6948243667342435, 0.6837357762650802, 0.6577653847707857, 0.6516622854564869, 0.6413600195415669, 0.629443010773553, 0.6162868349433923, 0.6075609393826766, 0.5976677358357853, 0.5922484555284315, 0.574315391737589, 0.566744739902959, 0.5556392079807803, 0.5451878601055541, 0.5370259918413288, 0.513840948367127, 0.5043578171639904, 0.4963434553897762, 0.48894830733527594, 0.47951420321659116, 0.45618192367140864, 0.43447387624022904, 0.4228157993446624, 0.4138031993071401, 0.40549527591015855, 0.3953643720753778, 0.38417870258324954, 0.3731515032553941, 0.3624899240307693, 0.3501964543591571, 0.3395782350233928, 0.32686320273691477, 0.31580769371342965, 0.3042645570131842, 0.2970915294801039, 0.27677519932074934, 0.2560411626195867, 0.2457969869530282, 0.23340020919030297, 0.22399656226631548, 0.20835902131639222, 0.19802602065849956, 0.1846316136278649, 0.16784201713881763, 0.15678810825157696, 0.14651700105058918, 0.13509939834360726, 0.12583543387268634, 0.11035577240606453, 0.09572585360723644, 0.08917314935436961, 0.0821466014934013, 0.07371103145347885, 0.06478356926778468, 0.05633291221081135, 0.04911181661069924, 0.04194001629912418, 0.03333469987682852, 0.020262720610771993, 0.010069121263729703]

plt.plot(x, base_y , color='red', label='base_50')
plt.plot(x, novel_y , color='green', label='novel_50')
plt.plot(x, optimal_result_y , color='blue', label='optimal result')
#plt.plot(x, fixed_novel_y , color='black', label='label fixed novel_50')
#plt.plot(x, filter_fp_novel_y , color='olive', label='filter fp novel_50')
#plt.plot(x, filter_fp_base_y , color='pink', label='filter fp base_50')

plt.plot(x, add_fn_novel_y , color='olive', label='add fn novel_50')
plt.plot(x, add_fn_base_y , color='pink', label='add fn base_50')

font1 = {'size': 10}
plt.legend(loc=4, prop=font1) # 显示图例
plt.grid()
plt.xlabel('recall', fontsize=13)
plt.ylabel('precision', fontsize=13)
#plt.ylabel('mAP(%)', fontsize=13)
plt.show()