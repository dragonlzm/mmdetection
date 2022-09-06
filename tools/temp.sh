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


# 200 clip proposal filpping(reg with class embedding, add, merge1, 2x regression loss, base filtered proposal, per bbox weight, v3, mlp)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge1'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 model.backbone.type='ResNetWithVitV3' model.backbone.merge_with_mlp=True"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48_merge1_v3_mlp"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    ${ADDITIONAL_CONFIG} \
    --resume-from=${WORK_DIR}/latest.pth

# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}


# 200 clip proposal filpping(reg with class embedding, add, merge2, 2x regression loss, base filtered proposal, per bbox weight, v3, mlp)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge2'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 model.backbone.type='ResNetWithVitV3' model.backbone.merge_with_mlp=True"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48_merge2_v3_mlp"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    ${ADDITIONAL_CONFIG} \
    --resume-from=${WORK_DIR}/latest.pth

# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}


# 200 clip proposal filpping(reg with class embedding, add, merge4, 2x regression loss, base filtered proposal, per bbox weight, v3, mlp)
ADDITIONAL_CONFIG="model.backbone.merge_step=['merge4'] model.rpn_head.loss_bbox.loss_weight=2.0 model.roi_head.bbox_head.loss_bbox.loss_weight=2.0 model.backbone.type='ResNetWithVitV3' model.backbone.merge_with_mlp=True"
WORK_DIR="/project/nevatia_174/zhuoming/detection/grad_clip_check/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48_merge4_v3_mlp"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_per_base_filtered_clip_proposal_weight_base48.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.temperature=100 model.train_cfg.rcnn.distill_loss_factor=1 optimizer_config.grad_clip.max_norm=10 \
    ${ADDITIONAL_CONFIG} \
    --resume-from=${WORK_DIR}/latest.pth


# test the model
#CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
CHECKPOINT_NAME="latest.pth"

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_base48.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_novel17.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/novel_results_trick \
--cfg-options model.roi_head.bbox_head.filter_base_cate=data/embeddings/base_finetuned_48cates.pt data.test.eval_filter_empty_gt=False \
data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_distillation_with_vit_bn65.py \
${WORK_DIR}/${CHECKPOINT_NAME} 2 --eval bbox segm \
--eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
${ADDITIONAL_CONFIG}


# with gt embeddings + 50 random embedding
WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_50_rand_embedding_over_sample_per_box_weight"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options runner.max_epochs=18 model.rpn_head.open_ln=True model.rpn_head.use_gt_name=True model.rpn_head.use_rand_name=50 model.rpn_head.use_size_weight=True \
    #--resume-from=${WORK_DIR}/latest.pth


# for testing
bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/base_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_novel_train.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/novel_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
${WORK_DIR}/latest.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/all_results
