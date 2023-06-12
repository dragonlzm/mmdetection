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

# Mask R-CNN with Distillation experiment on LVIS freq(base)/comm+rare(novel)

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data



# 24 epoch 
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"

# 24 epoch with hyper parameters change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_301212_learnable_temp"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"

# # 24 epoch with seesaw
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"


# # 24 epoch with seesaw with hyper-pararms change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"


# for rpn proposal_without finetuning
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 model.train_cfg.rcnn.use_only_clip_prop_for_distill=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_fc866.py"


#### 1x distillation 
# 24 epoch with hyper parameters change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_301212_learnable_temp_1xdist"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"

# 24 epoch with seesaw with hyper-pararms change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp_1xdist"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"


### 500 distillation bboxes
# 24 epoch with hyper parameters change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_301212_learnable_temp_1xdist_500b"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_500.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"

# # for rpn proposal_without finetuning
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 model.train_cfg.rcnn.use_only_clip_prop_for_distill=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405_500.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_fc866.py"


### 500 distillation bboxes + 2xreg + 1x dist
#24 epoch with hyper parameters change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_1xdist_2xreg_500b"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_500.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"


#### 500 distillation bboxes + 2xreg + 1x dist + base proposal dampen
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_1xdist_2xreg_500b_dampen_base"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_500_dampen_base.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866.py"


# 24 epoch with seesaw 2xreg 0.5 dist
ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_2xreg_05xdist"
TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_48e.py"
TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"

bash tools/new_dist_train.sh ${TRAIN_CONFIG} 2 \
${WORK_DIR} /data/zhuoming/detection \
--cfg-options ${ADDITIONAL_CONFIG} \
#--resume-from=${WORK_DIR}/latest.pth


# 48 epoch exp
# 48 epoch with seesaw with hyper-pararms change
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
# model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 \
# model.roi_head.bbox_head.num_shared_convs=3 model.roi_head.bbox_head.num_shared_fcs=0 \
# model.roi_head.bbox_head.num_cls_convs=1 model.roi_head.bbox_head.num_cls_fcs=2 \
# model.roi_head.bbox_head.num_reg_convs=1 model.roi_head.bbox_head.num_reg_fcs=2 \
# model.roi_head.bbox_head.learnable_temperature=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp_48e"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_48e.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_fc866_seesawloss.py"
# START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_freq_tuned_clipproposal_freq405_seesawloss_301212_learnable_temp/epoch_16.pth"

# for rpn proposal_without finetuning
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.temperature=100 optimizer_config.grad_clip.max_norm=10 model.train_cfg.rcnn.use_only_clip_prop_for_distill=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405_48e"
# TRAIN_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405_48e.py"
# TEST_CONFIG="configs/mask_rcnn_distill/mask_rcnn_distillation_lvis_raw_fc866.py"
# START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_raw_rpn_proposal_freq405/epoch_16.pth"



# bash tools/new_dist_train.sh ${TRAIN_CONFIG} 2 \
# ${WORK_DIR} /data/zhuoming/detection \
# --cfg-options ${ADDITIONAL_CONFIG} \
# --resume-from=${START_FROM}
# #--resume-from=${WORK_DIR}/latest.pth



CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh ${TEST_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}

