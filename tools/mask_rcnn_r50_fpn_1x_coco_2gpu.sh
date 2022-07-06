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

#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/test/mask_rcnn_r50_fpn_1x_coco_2gpu 
    #--resume-from=/project/nevatia_174/zhuoming/detection/test/mask_rcnn_r50_fpn_1x_coco_2gpu/latest.pth

# for 1x training
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48
#    --resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_base48/latest.pth

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
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm
    #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt
#     #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_r50_fpn_1x_coco_2gpu_clip_pretrain_base48_ori_norm_ori_opt/latest.pth