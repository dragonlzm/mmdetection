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

## this is for training the RPN baseline on LVIS dataset

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data


### 2x without see-saw loss
# all
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/rpn/rpn_r50_fpn_2x_lvis.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis \
   --cfg-options optimizer.lr=0.005 \
   #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis/latest.pth

# base
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/rpn/rpn_r50_fpn_2x_lvis_base.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis_base \
#    --cfg-options optimizer.lr=0.005 \
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis_base/latest.pth


# rpn_r50_fpn_2x_lvis_freq
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/rpn/rpn_r50_fpn_2x_lvis_freq.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis_freq \
#    --cfg-options optimizer.lr=0.005 \
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_2x_lvis_freq/latest.pth

