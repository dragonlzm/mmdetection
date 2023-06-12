#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=4:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

# this script for testing and finetuning of the attention RPN (only RPN, for the original version and the text-base version) 
# (paper: Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector)

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

EXTRA_CONFIG=""

# for image fewshot
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention-rpn_r50_c4_2xb2_coco_official-base-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_2xb2_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_240000.pth"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention-rpn_r50_c4_2xb4_coco_official-base-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_2xb4_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_120000.pth"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention-rpn_r50_c4_2xb4_coco_official-base48-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention-rpn_r50_c4_2xb4_coco_official-base48-training.py"
# CHECKPOINT_NAME="iter_120000.pth"


# for text zeroshot
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb2_coco_official-base-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb2_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_240000.pth"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_120000.pth"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# CHECKPOINT_NAME="iter_120000.pth"

WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm"
CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
CHECKPOINT_NAME="iter_120000.pth"
EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_with_norm"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# CHECKPOINT_NAME="iter_120000.pth"
# EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True"


# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_query"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_120000.pth"
# EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True --cfg-options model.rpn_head.linear_mapping=on_query"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_query"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# CHECKPOINT_NAME="iter_120000.pth"
# EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True --cfg-options model.rpn_head.linear_mapping=on_query"


# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_both"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# CHECKPOINT_NAME="iter_120000.pth"
# EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True --cfg-options model.rpn_head.linear_mapping=on_both"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_both"
# CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# CHECKPOINT_NAME="iter_120000.pth"
# EXTRA_CONFIG="--cfg-options model.rpn_head.normalize_img_feat=True --cfg-options model.rpn_head.linear_mapping=on_both"


# test the model on novel before the finetuning
bash tools/dist_fewshot_test.sh ${CONFIG_FILE} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/test_novel \
${EXTRA_CONFIG}

# finetune the model and get the performance after finetuning
bash tools/new_dist_fewshot_train.sh ${CONFIG_FILE} 2 \
${WORK_DIR}/finetuning /project/nevatia_174/zhuoming/detection \
${EXTRA_CONFIG} \
load_from=${WORK_DIR}/${CHECKPOINT_NAME}

# test the model on base after the finetuning
bash tools/dist_fewshot_test.sh ${BASE_CONFIG_FILE} \
${WORK_DIR}/finetuning/latest.pth 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/finetuning/rpn_base_finetuned \
${EXTRA_CONFIG}
