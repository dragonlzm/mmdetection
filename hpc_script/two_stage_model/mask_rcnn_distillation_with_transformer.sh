#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

# the exploration of using transformer in R-CNN head

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# base_filtered gt weight = 1
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# base_filtered gt weight = 1, v2
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_per_base_filtered_clip_proposal_weight_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# base_filtered gt weight = 1, v2, use_proposal_for_distill
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_dist_with_prop_per_base_filtered_clip_proposal_weight_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     model.train_cfg.rcnn.use_proposal_for_distill=True \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

### update lr 
# base_filtered gt weight = 1, v2, use_proposal_for_distill, tranformer_multiplier=0.01
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_dist_with_prop_per_base_filtered_clip_proposal_weight_base48_tm01"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     model.train_cfg.rcnn.use_proposal_for_distill=True \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# base_filtered gt weight = 1, v2, use_proposal_for_distill, tranformer_multiplier=0.005
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_dist_with_prop_per_base_filtered_clip_proposal_weight_base48_tm005"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     model.train_cfg.rcnn.use_proposal_for_distill=True optimizer.tranformer_multiplier=0.005 \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

### update grad clip
# base_filtered gt weight = 1, v2, use_proposal_for_distill, tranformer_multiplier=0.005, whole model 0.1
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_dist_with_prop_per_base_filtered_clip_proposal_weight_base48_tm005_gcw1"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 \
#     ${ADDITIONAL_CONFIG} \
#     model.train_cfg.rcnn.use_proposal_for_distill=True optimizer.tranformer_multiplier=0.005 optimizer_config.grad_clip.max_norm=0.1 \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# base_filtered gt weight = 1, v2, use_proposal_for_distill, tranformer_multiplier=0.005, whole model 0.01
# ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformerv2_dist_with_prop_per_base_filtered_clip_proposal_weight_base48_tm005_gcw01"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 \
#     ${ADDITIONAL_CONFIG} \
#     model.train_cfg.rcnn.use_proposal_for_distill=True optimizer.tranformer_multiplier=0.005 optimizer_config.grad_clip.max_norm=0.01 \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# base_filtered gt weight = 1, v2, use_proposal_for_distill, tranformer_multiplier=0.005
ADDITIONAL_CONFIG="model.roi_head.type='StandardRoIHeadDistillWithTransformerV2'"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_transformer_bs4_lr0005"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48_paramwise_grad_clip.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer.lr=0.005 \
    ${ADDITIONAL_CONFIG} optimizer_config.cumulative_iters=1 \
    model.train_cfg.rcnn.use_proposal_for_distill=True \
    #--resume-from=${WORK_DIR}/latest.pth
    #--seed=43 --deterministic \

# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG} \

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG} \

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG} \

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_transformer_per_base_filtered_clip_proposal_weight_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG} \