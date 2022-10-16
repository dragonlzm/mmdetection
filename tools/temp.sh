#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# base_filtered gt weight = 1, map after distill, linear
ADDITIONAL_CONFIG="model.roi_head.bbox_head.mapping_after_dist='linear'"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_map_after_dist_linear_1"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    ${ADDITIONAL_CONFIG} \
    #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, map after distill, mlp
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.mapping_after_dist='mlp'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_map_after_dist_linear_1"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG}
#     #--resume-from=${WORK_DIR}/latest.pth


# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True ${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True ${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True ${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True ${ADDITIONAL_CONFIG}



# train lvis v2 2xregw 1xdistw, fix set scale, 12 epoch
ADDITIONAL_CONFIG="model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_lvis_base_seesawloss_12e_2xreg_1xdistw_fixsetscale"
bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_base_seesawloss_12e.py 2 \
${WORK_DIR} ./data \
--cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
${ADDITIONAL_CONFIG} \
--resume-from=${WORK_DIR}/latest.pth


CHECKPOINT_NAME="latest.pth"
bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_lvis_bn_seesawloss.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel
