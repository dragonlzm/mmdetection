#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174
# for training the few-shot model attention_rpn

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# this script for base training of the attention RPN (paper: Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector)

# 2*2 batch size
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_2xb2_coco_official-base-training"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/attention-rpn_r50_c4_2xb2_coco_official-base-training.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_2xb4_coco_official-base-training"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
# 	configs/attention-rpn/attention-rpn_r50_c4_2xb4_coco_official-base-training.py --launcher pytorch \
# 	--work-dir=${WORK_DIR} \
# 	#--resume-from=${WORK_DIR}/latest.pth

# 4*2 batch size base48
WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_2xb4_coco_official-base48-training"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/few_shot_train.py \
	configs/attention-rpn/attention-rpn_r50_c4_2xb4_coco_official-base48-training.py --launcher pytorch \
	--work-dir=${WORK_DIR} \
	#--resume-from=${WORK_DIR}/latest.pth