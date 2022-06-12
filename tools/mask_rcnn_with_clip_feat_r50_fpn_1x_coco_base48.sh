#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=35GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# for 100 random proposal raw feat
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_old_embed_repro \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
    #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48/latest.pth

# for 200 clip propsal raw feat
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
#     #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200clip_pro/latest.pth

# for 200 random proposal raw feat
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10
#     #--resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro/latest.pth

# for 200 random proposal raw feat 2x
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_2x_coco_base48_200random_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_2x_coco_base48_200random_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_1x_coco_base48_200random_pro/epoch_8.pth

# for 200 random proposal raw feat 3x
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_3x_coco_base48_200random_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_3x_coco_base48_200random_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_2x_coco_base48_200random_pro/epoch_16.pth

# for 200 random proposal raw feat 4x
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_with_clip_feat_r50_fpn_4x_coco_base48_200random_pro.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_4x_coco_base48_200random_pro \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_with_clip_feat_r50_fpn_3x_coco_base48_200random_pro/epoch_24.pth


