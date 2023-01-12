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

# train lvis v1 range scale 1x
# ADDITIONAL_CONFIG="model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_seesawloss.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# ${ADDITIONAL_CONFIG} \
# #--resume-from=${WORK_DIR}/latest.pth

# train lvis v1 range scale 2x
# START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss"
# ADDITIONAL_CONFIG="model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss_24e"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_seesawloss_24e.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# ${ADDITIONAL_CONFIG} \
# --resume-from=${START_FROM}/epoch_8.pth
# #--resume-from=${WORK_DIR}/latest.pth

# # train lvis v1 range scale 3x
# START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss_24e"
# ADDITIONAL_CONFIG="model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss_36e"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_seesawloss_36e.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# ${ADDITIONAL_CONFIG} \
# --resume-from=${START_FROM}/epoch_16.pth
# #--resume-from=${WORK_DIR}/latest.pth

# 48 epoch 2xreg 0.5 distill 302121
ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss_48e_2xregw_05distw_301212"

bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_base_seesawloss_48e.py 2 \
${WORK_DIR} /data/zhuoming/detection \
--cfg-options ${ADDITIONAL_CONFIG} \
--resume-from=${WORK_DIR}/epoch_1.pth
#--resume-from=${WORK_DIR}/latest.pth


CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}

