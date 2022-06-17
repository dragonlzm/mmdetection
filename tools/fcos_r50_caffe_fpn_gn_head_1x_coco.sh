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

# 80 cates
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco \
   --cfg-options optimizer.lr=0.0025
   #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/latest.pth

# 48 cates
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48 \
   --cfg-options optimizer.lr=0.0025
   #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48/latest.pth