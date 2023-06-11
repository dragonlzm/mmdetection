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

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
# WORK_DIR="workdir1"
# ROOT="/project/nevatia_174/zhuoming/detection/coco/clip_proposal_feat/"

# cd ${ROOT}
# mkdir ${WORK_DIR}
# cd ${WORK_DIR}
# git clone -b new_rpn https://github.com/dragonlzm/mmdetection
# cd mmdetection
# rm -rf ./data
# ln -sf /project/nevatia_174/zhuoming/detection ./data


CONFIG_FILE="configs/cls_finetuner/cls_finetuner_clip_full_coco.py"
CHECKPOINT="data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth"
FEAT_SAVE_PATH_ROOT="data/coco/clip_proposal_feat/base48_finetuned_channel_corr"
#BBOX_SAVE_PATH_ROOT="data/coco/clip_proposal/32_32_512"
BBOX_SAVE_PATH_ROOT="data/coco/clip_proposal/32_32_512_channel_corr"

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

bash tools/dist_test.sh ${CONFIG_FILE} \
${CHECKPOINT} 2 --eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned/add_conf \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
   model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
   model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True

# cd ..
# rm -rf mmdetection