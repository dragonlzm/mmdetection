#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# rn50 initialization
# WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_4xb2_coco_official-base-training_2gpu"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/attention-rpn/attention-rpn_r50_c4_4xb2_coco_official-base-training_2gpu.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

WORK_DIR="/project/nevatia_174/zhuoming/detection/meta_learning/attention-rpn_r50_c4_4xb2_coco_official-base-training_2gpu_bs8"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/attention-rpn/attention-rpn_r50_c4_4xb2_coco_official-base-training_2gpu_bs8.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth