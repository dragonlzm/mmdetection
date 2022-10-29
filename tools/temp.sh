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

# train lvis v2 2xregw 1xdistw, fix set scale, 24 epoch (merge 3)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_seesawloss_24e_2xreg_1xdistw_fixsetscale_merge3"
CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}



# test lvis v2 3xregw 0.5distw, range scale, 12 epoch (merge 3)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_seesawloss_24e_3xreg_05xdistw_rangescale_merge3"
CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}


# train lvis v2 2xregw 1xdistw, fix set scale, 12 epoch (merge 3)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_seesawloss_12e_2xreg_1xdistw_fixsetscale"
CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}



# train lvis v2 3xregw 0.5distw, range scale, 12 epoch (merge 3)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge3'] model.rpn_head.loss_bbox.loss_weight=3.0 model.roi_head.bbox_head.loss_bbox.loss_weight=3.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_lvis_base_seesawloss_24e_3xreg_05xdistw_rangescale"
CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options ${ADDITIONAL_CONFIG}
