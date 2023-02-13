#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
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

## this script aim to generate feat for the LVIS dataset with the clip proposal generated for LVIS
## due to the long-tail property of the LVIS dataset, we can expect that we should not filter the base cate in the dataset
## since the novel categories only exist in about 300 images, and the whole dataset have nearly 100,000 images
## this ratio make filtering the base categories does not make sense.

# spliting the generation into other section to accelerate the procedure
# CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2/epoch_18.pth"
# CONFIG_FILE="configs/cls_finetuner/cls_finetuner_clip_full_lvis.py"
# BBOX_SAVE_PATH_ROOT="data/lvis_v1/rpn_proposal/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base"
# FEAT_SAVE_PATH_ROOT="data/lvis_v1/rpn_proposal_feat/lvis_base_finetuned"

# bash tools/dist_test.sh \
# configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
# data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2/epoch_18.pth 2 \
# --eval=gt_acc \
# --eval-options jsonfile_prefix=data/lvis_v1/rpn_proposal_feat/lvis_base_finetuned/extract_feat \
# --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
# model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/lvis_v1/rpn_proposal_feat/lvis_base_finetuned \
# model.test_cfg.use_pregenerated_proposal=data/lvis_v1/rpn_proposal/mask_rcnn_r50_fpn_random_seesaw_loss_normed_mask_mstrain_2x_lvis_v1_base \
# model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True


# generate rpn proposal
bash tools/dist_test.sh \
configs/rpn/rpn_r50_fpn_1x_lvis.py \
data/exp_res/mask_rcnn_r50_fpn_sample1e-3_mstrain_2x_lvis_v1_freq/epoch_24.pth 2 \
--out rpn_r50_fpn_1x_lvis.pkl \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json data.test.img_prefix=data/lvis_v1/ \
model.test_cfg.bbox_save_path_root=data/lvis_v1/rpn_proposal_1/mask_rcnn_freq/

## raw clip model, rpn proposal
CHECKPOINT="data/test/cls_finetuner_clip_base_100shots_train/epoch_0.pth"
CONFIG_FILE="configs/cls_finetuner/cls_finetuner_clip_full_lvis.py"
BBOX_SAVE_PATH_ROOT="data/lvis_v1/rpn_proposal_1/mask_rcnn_freq"
FEAT_SAVE_PATH_ROOT="data/lvis_v1/rpn_proposal_feat/freq_proposal_raw_feat"

bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/lvis_v1/rpn_proposal_feat/freq_proposal_raw_feat/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True

