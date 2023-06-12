#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

# generate the feature for LVIS dataset (zero-shot setting), using the CLIP porposal

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

## this script aim to generate feat for the LVIS dataset with the clip proposal generated for LVIS
## due to the long-tail property of the LVIS dataset, we can expect that we should not filter the base cate in the dataset
## since the novel categories only exist in about 300 images, and the whole dataset have nearly 100,000 images
## this ratio make filtering the base categories does not make sense.

# 1.proposal location
# 2.used model for feat extraction
# 3.the write path



## generate imagenet1762_finetuned feature
#1
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only'

#2
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#3
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#4
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#5
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 


#6
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 


#7
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#8
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 




#9
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#10
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#11
bash tools/dist_test_23.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 




#12
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only'  

#13
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#14
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 

#15
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=True model.base_cate_name='coco_base_only' 


# remain
bash tools/dist_test_23_double.sh \
configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/imagenet1762_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512_imagenet1762 \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
data.test.eval_filter_empty_gt=False model.base_cate_name='coco_base_only' 
