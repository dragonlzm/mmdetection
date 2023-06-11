#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/generate_feat/mmdetection
# WORK_DIR="workdir1"
# ROOT="/project/nevatia_174/zhuoming/detection/coco/clip_proposal_feat/"

# cd ${ROOT}
# mkdir ${WORK_DIR}
# cd ${WORK_DIR}
# git clone -b generate_feat https://github.com/dragonlzm/mmdetection
# cd mmdetection
# rm -rf ./data
# ln -sf /project/nevatia_174/zhuoming/detection ./data

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_500_1xrandom_sorted \
   model.test_cfg.use_pregenerated_proposal=data/coco/clip_proposal/32_32_512 \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

# cd ..
# rm -rf mmdetection