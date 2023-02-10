#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
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
#CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2/epoch_18.pth"
CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_freq405_train_gt_only_100_rand/epoch_18.pth"
CONFIG_FILE="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_lvis_fc866.py"
BBOX_SAVE_PATH_ROOT="data/detection/lvis_v1/clip_proposal/lvis_fc866_32_32_512"
#BBOX_SAVE_PATH_ROOT="data/lvis_v1/clip_proposal/lvis_32_32_512"
FEAT_SAVE_PATH_ROOT="data/lvis_v1/clip_proposal_feat/lvis_freq_finetuned_vision_and_text"
#FEAT_SAVE_PATH_ROOT="data/lvis_v1/clip_proposal_feat/lvis_base_finetuned"

#### update for using the best overall perf model to extract feature, no longer filter the base cate
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True \
model.test_cfg.generate_gt_feat=True


# # 1
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_0_8000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 2
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_8000_16000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 3
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_16000_24000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 4
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_24000_32000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 5
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_32000_40000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 6
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_40000_48000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True

# # 7
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    ${CONFIG_FILE} \
#    ${CHECKPOINT} \
#    --launcher pytorch --eval=gt_acc \
#    --eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
#    --cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train_48000_56000.json \
#    model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
#    model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
#    model.test_cfg.num_of_rand_bboxes=1000 model.test_cfg.filter_clip_proposal_base_on_cates=True \
#    model.test_cfg.save_cates_and_conf=True
