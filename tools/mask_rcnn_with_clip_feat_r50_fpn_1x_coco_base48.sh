#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=36:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# for 1x training
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48
    #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_base48/latest.pth

# for 1x training
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/reimplement_distillation/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48
#     #--resume-from=/project/nevatia_174/zhuoming/detection/reimplement_distillation/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48 \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
#     #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48/latest.pth


# for 200 clip propsal raw feat
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
    #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro/latest.pth

# for 200 random proposal raw feat
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
#     #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro/latest.pth