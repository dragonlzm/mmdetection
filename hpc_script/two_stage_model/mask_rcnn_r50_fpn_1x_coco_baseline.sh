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

# training the Mask R-CNN baseline without distillation

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# all 80 baseline
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_correct_lr \
#     #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_correct_lr/latest.pth

# base 48 baseline
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_base48
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/latest.pth

# # novel 17 baseline
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17_reg_class_agno
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17_reg_class_agno/latest.pth


# all 80 baseline(class specific)
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_reg_class_spec \
#     --cfg-options model.roi_head.bbox_head.reg_class_agnostic=False \
#     #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_reg_class_spec/latest.pth

# base 48 baseline(class specific)
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_base48_reg_class_spec \
#    --cfg-options model.roi_head.bbox_head.reg_class_agnostic=False \
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_base48_reg_class_spec/latest.pth

# novel 17 baseline(class specific)
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17_reg_class_spec \
   --cfg-options model.roi_head.bbox_head.reg_class_agnostic=False \
   #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_2gpu_novel17_reg_class_spec/latest.pth

# for 2*2 novel17 (should be delete in this script)
# TRAIN_CONFIG="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17_reg_class_spec"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.reg_class_agnostic=False \
#     #--resume-from=${WORK_DIR}/latest.pth  

# for 2*2 novel17 class agno (should be delete in this script)
# TRAIN_CONFIG="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17_reg_class_agno"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.reg_class_agnostic=True \
#     --resume-from=${WORK_DIR}/latest.pth  


# for 2x training first stage
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48 \
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/epoch_8.pth

# for clip feature
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/latest.pth

# for clip feature
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/latest.pth

# for vild baseline
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_vild_baseline.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_vild_baseline
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_vild_baseline/latest.pth

# for the new norm test
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm
#     #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt
#     #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt/latest.pth


# for 3x training
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_3x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_3x_coco_2gpu_base48 \
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_2x_coco_2gpu_base48/epoch_16.pth


# for 4x training
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_4x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_4x_coco_2gpu_base48 \
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_3x_coco_2gpu_base48/epoch_24.pth


# mask rcnn 3x base48 baseline
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_base48
#    #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_base48/latest.pth

# mask rcnn 4x base48 baseline
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_4x_coco_base48.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_mstrain-poly_4x_coco_base48 \
   --resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_base48/epoch_9.pth