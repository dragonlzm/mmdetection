#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
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
WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
	configs/attention-rpn/rpn_attention_text_rpn_r50_c4_2xb4_coco_official-base-training_with_norm.py --launcher pytorch \
	--work-dir=${WORK_DIR} \
	#--resume-from=${WORK_DIR}/latest.pth

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