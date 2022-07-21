#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# 2*2 batch size
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb2_coco_official-base-training"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb2_coco_official-base-training.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth


# 4*2 batch size (with norm)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (with norm)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_with_norm"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth


# 4*2 batch size (with norm, map on query)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_query"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	--cfg-options model.rpn_head.linear_mapping='on_query' \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (with norm, map on query)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_query"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	--cfg-options model.rpn_head.linear_mapping='on_query' \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size (with norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_both"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	--cfg-options model.rpn_head.linear_mapping='on_both' \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (with norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_both"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_with_norm.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	--cfg-options model.rpn_head.linear_mapping='on_both' \
# 	#--resume-from=${WORK_DIR}/latest.pth


# 4*2 batch size (w/o any norm, map on query)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_query_wo_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_query"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (w/o any norm, map on query)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_query_wo_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_query"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size (w/o any norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_both_wo_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_both"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (w/o any norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_both_wo_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_both"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth



# 4*2 batch size (with support norm, map on query)
WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_query_w_sup_norm"
FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_query model.rpn_head.normalize_text_feat=True"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
	${BASE_CONFIG_FILE} --launcher pytorch \
	--work-dir=${WORK_DIR} \
	${EXTRA_CONFIG} \
	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (with support norm, map on query)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_query_w_sup_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_query model.rpn_head.normalize_text_feat=True"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size (with support norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_map_on_both_w_sup_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_both model.rpn_head.normalize_text_feat=True"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48 (with support norm, map on both)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training_map_on_both_w_sup_norm"
# FINETUNING_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_coco_official-10shot-fine-tuning-novel17.py"
# BASE_CONFIG_FILE="configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base48-training.py"
# EXTRA_CONFIG="--cfg-options model.rpn_head.linear_mapping=on_both model.rpn_head.normalize_text_feat=True"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	${BASE_CONFIG_FILE} --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	${EXTRA_CONFIG} \
# 	#--resume-from=${WORK_DIR}/latest.pth


# testing and finetuning
CHECKPOINT_NAME="iter_120000.pth"

## test the model on novel before the finetuning
bash tools/dist_fewshot_test.sh ${FINETUNING_CONFIG_FILE} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/test_novel \
${EXTRA_CONFIG}

## finetune the model and get the performance after finetuning
bash tools/new_dist_fewshot_train.sh ${FINETUNING_CONFIG_FILE} 2 \
${WORK_DIR}/finetuning /project/nevatia_174/zhuoming/detection \
${EXTRA_CONFIG} \
load_from=${WORK_DIR}/${CHECKPOINT_NAME}

## test the model on base after the finetuning
bash tools/dist_fewshot_test.sh ${BASE_CONFIG_FILE} \
${WORK_DIR}/finetuning/latest.pth 2 --eval proposal_fast \
--eval-options jsonfile_prefix=${WORK_DIR}/finetuning/rpn_base_finetuned \
${EXTRA_CONFIG}