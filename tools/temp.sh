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

### fixed sample
# for mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/seesaw_loss/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base_sample_fixed \
   --cfg-options optimizer.lr=0.005 \
   --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base_sample_fixed/latest.pth

# for mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_base
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_base.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_base_sample_fixed \
   --cfg-options optimizer.lr=0.005 evaluation.metric='bbox' \
   --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_base_sample_fixed/latest.pth