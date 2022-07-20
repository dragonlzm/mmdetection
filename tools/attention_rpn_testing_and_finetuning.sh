#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=2:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# for attention rpn detector
WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_2xb2_coco_official-base-training"
CHECKPOINT_NAME="iter_240000.pth"
FINETUNE_CONFIG_FILE="configs/attention-rpn/attention-rpn_r50_c4_coco_official-10shot-fine-tuning.py"
BASE_CONFIG_FILE="configs/attention-rpn/attention-rpn_r50_c4_2xb2_coco_official-base-training.py"

# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_2xb4_coco_official-base-training"
# CHECKPOINT_NAME="iter_120000.pth"
# FINETUNE_CONFIG_FILE="configs/attention-rpn/attention-rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/attention-rpn_r50_c4_2xb4_coco_official-base-training.py"

RPN_NOVEL_CONFIG="rpn_attention-rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# it doesn't matter which rpn config to test the more here, since the model structures are the same.
RPN_BASE_CONFIG="rpn_attention-rpn_r50_c4_2xb2_coco_official-base-training.py"

# test the model on novel before the finetuning (here we get the novel ap)
# bash tools/dist_fewshot_test.sh ${FINETUNE_CONFIG_FILE} \
# ${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval proposal_fast \
# --eval-options jsonfile_prefix=${WORK_DIR}/test_novel

# finetune the model and get the performance after finetuning (here we get the novel ap)
# bash tools/new_dist_fewshot_train.sh ${FINETUNE_CONFIG_FILE} 2 \
# ${WORK_DIR}/finetuning /project/nevatia_174/zhuoming/detection --cfg-options \
# load_from=${WORK_DIR}/${CHECKPOINT_NAME}

# test the model on base after the finetuning (here we get the novel ap)
# bash tools/dist_fewshot_test.sh ${BASE_CONFIG_FILE} \
# ${WORK_DIR}/finetuning/latest.pth 2 --eval proposal_fast \
# --eval-options jsonfile_prefix=${WORK_DIR}/finetuning/rpn_base_finetuned


# test the RPN on novel before finetune 
bash tools/dist_fewshot_test.sh ${RPN_NOVEL_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/test_rpn_novel

# test the RPN on base before finetune 
bash tools/dist_fewshot_test.sh ${RPN_BASE_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/test_rpn_base

# test the RPN on novel after finetune 
bash tools/dist_fewshot_test.sh ${RPN_NOVEL_CONFIG} \
${WORK_DIR}/finetuning/latest.pth 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/finetuning/test_rpn_novel

# test the RPN on base after finetune 
bash tools/dist_fewshot_test.sh ${RPN_BASE_CONFIG} \
${WORK_DIR}/finetuning/latest.pth 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/finetuning/test_rpn_novel
