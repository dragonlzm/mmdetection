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

# this script is for training the deformable detr 

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# 80 cates
# bash tools/new_dist_train.sh configs/deformable_detr/deformable_detr_r50_2x2_50e_coco_grad8xcumulated.py 2 \
# data/one_stage/deformable_detr_r50_2x2_50e_coco_grad8xcumulated /data/zhuoming/detection \
# #--resume-from=data/one_stage/retinanet_r50_fpn_1x_coco_base48/latest.pth

# 48 cates
bash tools/new_dist_train.sh configs/deformable_detr/deformable_detr_r50_2x2_50e_coco_grad8xcumulated_base48.py 2 \
data/one_stage/deformable_detr_r50_2x2_50e_coco_grad8xcumulated_base48 /data/zhuoming/detection \
#--resume-from=data/one_stage/deformable_detr_r50_2x2_50e_coco_grad8xcumulated_base48/latest.pth