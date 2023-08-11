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

# exploration of using the ViT backbone from CLIP to train the Mask R-CNN with distillation
# Version2, the merging method is different

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# 200 clip proposal filpping(reg with class embedding, cat, with vit backbone v2, 8*8 grids)
EXTRA_CONFIG='model.backbone.image_grids_num=[8,8]'
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_v2_base48_8x8_grids"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_base48.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    ${EXTRA_CONFIG} \
    #--resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(reg with class embedding, cat, with vit backbone v2, 6*6 grids)
# EXTRA_CONFIG='model.backbone.image_grids_num=[6,6]'
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_v2_base48_6x6_grids"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${EXTRA_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# 200 clip proposal filpping(reg with class embedding, cat, with vit backbone v2, 4*4 grids)
# EXTRA_CONFIG='model.backbone.image_grids_num=[4,4]'
# WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_v2_base48_4x4_grids"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     ${EXTRA_CONFIG} \
#     #--resume-from=${WORK_DIR}/latest.pth

# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${EXTRA_CONFIG} 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${EXTRA_CONFIG} 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${EXTRA_CONFIG} 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_v2_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${EXTRA_CONFIG}