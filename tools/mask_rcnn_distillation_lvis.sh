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

# base_filtered gt weight = 1 
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth


# base_filtered gt weight = 1, 2x regression weight
# ADDITIONAL_CONFIG="model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_regw2"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 0.5
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_distw05"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, 48epoch
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_48e"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_48e.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base/epoch_16.pth
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, 12epoch
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_12e"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_12e.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base/epoch_8.pth
    #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, 12epoch, use range scale
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_12e_range_scale"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_12e_range_scale.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    #--resume-from=${WORK_DIR}/latest.pth


# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_bn.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel

