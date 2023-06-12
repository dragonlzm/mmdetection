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

# exploration of using the ViT backbone from CLIP to train the Mask R-CNN with distillation
# on LVIS dataset


cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     --resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight, with ln open)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_lnopen"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     model.vit_backbone_cfg.open_ln=True \
#     --resume-from=${WORK_DIR}/latest.pth

### hyper-parameters tuning
# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight, range scale)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight, range scale, 0.5 distw)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale_distw05"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight, range scale, 0.5 distw， 3x reg)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale_distw05_regw3"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight, range scale, 0.5 distw, gt only)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale_distw05_gtonly"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.use_only_gt_pro_for_distill=True \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

### 2x4 experiment
# 200 clip proposal filpping(merge3, 2x regression loss, base filtered proposal, per bbox weight)
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_12e_2x4"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_12e.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     data.samples_per_gpu=4 optimizer.lr=0.01 \
#     #--resume-from=${WORK_DIR}/latest.pth

### v2 48 epochs
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_48e"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_48e.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     --resume-from=${WORK_DIR}/latest.pth
#     # --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base/epoch_16.pth


#### 3x experiment
### v2 36e
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_36e"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_36e.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_48e/epoch_24.pth
#     #--resume-from=${WORK_DIR}/latest.pth

### v2 36e range scale, 2x reg
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale_36e"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale_36e.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     --resume-from=/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale/epoch_16.pth
#     #--resume-from=${WORK_DIR}/latest.pth

### v2 36e range scale, 3x reg, 0.5distillw 
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_range_scale_36e_distw05_regw3"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_range_scale_36e.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
    ${ADDITIONAL_CONFIG} \
    --resume-from=${WORK_DIR}/latest.pth

### 3xregw, 0.5distw, fix set scale
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_36e_3xregw_05xdistw"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_36e.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#  ${ADDITIONAL_CONFIG} \
# #--resume-from=${WORK_DIR}/latest.pth

### 3xregw, 0.5distw, fix set scale, open ln
# ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0 model.vit_backbone_cfg.open_ln=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_36e_3xregw_05xdistw_lnopen"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_36e.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.5 optimizer_config.grad_clip.max_norm=10 \
#  ${ADDITIONAL_CONFIG} \
# #--resume-from=${WORK_DIR}/latest.pth




# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}