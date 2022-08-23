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

# base_filtered gt weight = 1 (reproduce, fixed seed)
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_seed43"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    --seed=43 --deterministic \
    #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, 2x distillation
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_2xdistw"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=2 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1, 3x setting
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# # base_filtered gt weight = 1, 3x setting, 2x distillation
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule_2xdistw"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=2 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 10
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_gtw10"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.gt_bboxes_distill_weight=10.0 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base not filtered gt weight = 1
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_clip_proposal_weight"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base not filtered gt weight = 10
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_clip_proposal_weight_gtw10"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.gt_bboxes_distill_weight=10.0 \
#     #--resume-from=${WORK_DIR}/latest.pth


# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.reg_with_cls_embedding=True 