#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# mask rcnn 1x baseline proposal
# COMBINE_METHOD='cat'
# TRAIN_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_base48_trained_1x_cls_agno_train2017.pkl'
# TEST_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_base48_trained_1x_cls_agno_val2017.pkl'

# WORK_DIR="/project/nevatia_174/zhuoming/detection/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
#     #--resume-from=${WORK_DIR}/latest.pth

# mask rcnn 2x baseline proposal
COMBINE_METHOD='cat'
TRAIN_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_base48_trained_2x_cls_agno_train2017.pkl'
TEST_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_base48_trained_2x_cls_agno_val2017.pkl'

WORK_DIR="/project/nevatia_174/zhuoming/detection/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48__baseline2x_proposal"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48__baseline2x_proposal.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
    #--resume-from=${WORK_DIR}/latest.pth

# mask rcnn distill flip, reg with embedding
# COMBINE_METHOD='cat'
# TRAIN_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_distill_base48_trained_1x_flip_reg_with_embed_train2017.pkl'
# TEST_PROPOSAL_FILE='data/coco/proposals/mask_rcnn_distill_base48_trained_1x_flip_reg_with_embed_val2017.pkl'

# WORK_DIR="/project/nevatia_174/zhuoming/detection/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48__distill_filp_reg_with_embed_proposal"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48__distill_filp_reg_with_embed_proposal.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
#     model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
#     #--resume-from=${WORK_DIR}/latest.pth


# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
data.val.proposal_file=${TEST_PROPOSAL_FILE}

bash tools/dist_test.sh configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
data.val.proposal_file=${TEST_PROPOSAL_FILE}

bash tools/dist_test.sh configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
data.val.proposal_file=${TEST_PROPOSAL_FILE}

bash tools/dist_test.sh configs/mask_rcnn_distill_rc_decouple/mask_rcnn_r50_fpn_with_clip_feat_rc_decouple_1x_coco_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.roi_head.bbox_head.combine_reg_and_cls_embedding=${COMBINE_METHOD} \
data.val.proposal_file=${TEST_PROPOSAL_FILE}