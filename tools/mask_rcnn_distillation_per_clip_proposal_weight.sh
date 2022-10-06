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
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_seed43"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --seed=43 --deterministic \
#     #--resume-from=${WORK_DIR}/latest.pth

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


# base_filtered gt weight = 1, 4x setting
# START_FROM="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_4xschedule"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_4xschedule.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --resume-from=${START_FROM}/epoch_9.pth

### 300 proposal exps
# base_filtered gt weight = 1 (reproduce, fixed seed, 1.0 distillation weight)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     --seed=42 --deterministic \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1 (reproduce, fixed seed, 2/3 distillation weight)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300_distw0666"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.666 optimizer_config.grad_clip.max_norm=10 \
#     --seed=42 --deterministic \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1 (reproduce, fixed seed, 1.0 distillation weight, unfixed)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300_unfixed"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt weight = 1 (reproduce, fixed seed, 2/3 distillation weight, unfixed)
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300_distw0666_unfixed"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_300.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=0.666 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

ADDITIONAL_CONFIG = ""
### regression with mlp
# base_filtered gt weight = 1 (reproduce, fixed seed)
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.reg_with_mlp=True"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_reg_with_mlp"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

### per bbox weight
# sqrt version
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.per_bbox_reg_weight='sqrt'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_per_size_reg_weight_v1"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# ori version
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.per_bbox_reg_weight='ori'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_per_size_reg_weight_v2"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \

# log version
# ADDITIONAL_CONFIG="model.roi_head.bbox_head.per_bbox_reg_weight='log'"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_per_size_reg_weight_v3"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${ADDITIONAL_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth
#     #--seed=43 --deterministic \


#### with mix gt feature
# base_filtered 
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_mix_gt"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_mix_gt.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     #--resume-from=${WORK_DIR}/latest.pth

# base_filtered gt only
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_mix_gt_gt_only"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_mix_gt.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.use_only_gt_pro_for_distill=True \
#     #--resume-from=${WORK_DIR}/latest.pth

#### clip proposal only
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_clip_proposal_only"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
# model.train_cfg.rcnn.use_only_clip_prop_for_distill=True \
# --resume-from=${WORK_DIR}/epoch_3.pth

#### gt only + damp factor
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_gt_only_with_damped_factors"
# bash tools/new_dist_train.sh configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight.py 2 \
# ${WORK_DIR} ./data \
# --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
# model.train_cfg.rcnn.use_only_gt_pro_for_distill=True model.train_cfg.rcnn.gt_only_damp_factor=True \
# #--resume-from=${WORK_DIR}/latest.pth


#### base_filtered per bbox weight with new rpn
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_new_rpn"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_new_rpn.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    #--resume-from=${WORK_DIR}/latest.pth


#### perturbation exp 3x
### gt only 3x perturbation
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule_gt_only_perturb"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.use_only_gt_pro_for_distill=True model.train_cfg.rcnn.add_distill_pertrub=True \
#     #--resume-from=${WORK_DIR}/latest.pth

### gt + clip proposal 3x perturbation
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule_perturb"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_per_base_filtered_clip_proposal_weight_3xschedule.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.train_cfg.rcnn.add_distill_pertrub=True \
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