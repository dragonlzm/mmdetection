#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=24:00:00
#SBATCH --account=nevatia_174

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# open all para in the rn backbone
WORK_DIR="/project/nevatia_174/zhuoming/detection/exp_res/cls_finetuner_clip_base48_all_train_resnet50"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_base48_all_train_resnet50.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth

# open only the norm layer in the rn backbnone
WORK_DIR="/project/nevatia_174/zhuoming/detection/exp_res/cls_finetuner_clip_base48_all_train_resnet50_only_bn"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_base48_all_train_resnet50_only_bn.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth