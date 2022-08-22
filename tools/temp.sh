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


# for mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/seesaw_loss/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1 \
   --cfg-options optimizer.lr=0.005 \
   --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1/latest.pth

# for mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1 \
   --cfg-options optimizer.lr=0.005 \
   --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1/latest.pth

