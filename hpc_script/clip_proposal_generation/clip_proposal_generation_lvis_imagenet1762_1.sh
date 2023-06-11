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

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

CONFIG_FILE="configs/cls_proposal_generator/cls_proposal_generator_lvis_imagenet1762_name.py"
JSONFILE_PREFIX="data/test/cls_proposal_generator_coco/results_lvis_32_32_512"
BBOX_SAVE_PATH_ROOT="data/detection/lvis_v1/clip_proposal/lvis_32_32_512_imagenet1762"
CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_100_rand_embedding_v2/epoch_18.pth"

# 1
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=proposal_fast \
--eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
--cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_0_8000.json model.test_cfg.use_sigmoid_for_cos=True

# # 2
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_8000_16000.json model.test_cfg.use_sigmoid_for_cos=True

# # 3
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_16000_24000.json model.test_cfg.use_sigmoid_for_cos=True

# # 4
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_24000_32000.json model.test_cfg.use_sigmoid_for_cos=True


# # 5
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_32000_40000.json model.test_cfg.use_sigmoid_for_cos=True


# # 6
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_40000_48000.json model.test_cfg.use_sigmoid_for_cos=True


# # 7
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_48000_56000.json model.test_cfg.use_sigmoid_for_cos=True


# # 8
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_56000_64000.json model.test_cfg.use_sigmoid_for_cos=True


# # 9
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_64000_72000.json model.test_cfg.use_sigmoid_for_cos=True


# # 10
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_72000_80000.json model.test_cfg.use_sigmoid_for_cos=True


# # 11
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_80000_88000.json model.test_cfg.use_sigmoid_for_cos=True


# # 12
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_88000_96000.json model.test_cfg.use_sigmoid_for_cos=True

# # 13
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_96000_104000.json model.test_cfg.use_sigmoid_for_cos=True


# # the remain
# bash tools/dist_test.sh \
# ${CONFIG_FILE} \
# ${CHECKPOINT} 2 \
# --eval=proposal_fast \
# --eval-options jsonfile_prefix=${JSONFILE_PREFIX}_1 \
# --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
# model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=False model.test_cfg.bbox_save_path_root=${BBOX_SAVE_PATH_ROOT} \
# data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json model.test_cfg.use_sigmoid_for_cos=True