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

# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embedding only
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_only"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_gt_only.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embeddings + 50 random embedding
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_50_rand_embedding"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_gt_only.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.rpn_head.use_rand_name=50 \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embeddings + 100 random embedding
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_100_rand_embedding"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_gt_only.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.rpn_head.use_rand_name=100 \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embeddings + 200 random embedding
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_200_rand_embedding"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_gt_only.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.rpn_head.use_rand_name=200 \
#     #--resume-from=${WORK_DIR}/latest.pth


### oversample experiments
# vision only
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embeddings + 50 random embedding
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_50_rand_embedding_over_sample"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.rpn_head.open_ln=True model.rpn_head.use_gt_name=True model.rpn_head.use_rand_name=50 \
#     #--resume-from=${WORK_DIR}/latest.pth

# with gt embeddings + 100 random embedding
# WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_100_rand_embedding_over_sample"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     --cfg-options model.rpn_head.open_ln=True model.rpn_head.use_gt_name=True model.rpn_head.use_rand_name=100 \
#     #--resume-from=${WORK_DIR}/latest.pth

### longer training schedule exp
# vision only(with lr drop)
START_FROM="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample"
WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v1"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options lr_config.step=[12,] runner.max_epochs=18 \
    --resume-from=${START_FROM}/latest.pth

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/base_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_novel_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/novel_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/all_results

# vision only(without lr drop)
START_FROM="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample"
WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options runner.max_epochs=18 \
    --resume-from=${START_FROM}/latest.pth

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/base_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_novel_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/novel_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/all_results



# bash tools/new_dist_train.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_over_sample.py 2 \
# data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample ./data \
# --cfg-options lr_config.step=[12,] runner.max_epochs=18 \
# --resume-from=data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample/epoch_12.pth

# bash tools/new_dist_train_23.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train_gt_only.py 2 \
# data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_100_rand_embedding ./data \
# --cfg-options lr_config.step=[12,] runner.max_epochs=18 model.rpn_head.use_rand_name=100 \
# --resume-from=data/exp_res/cls_finetuner_clip_lvis_base_train_gt_and_100_rand_embedding/epoch_12.pth

# bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
# ${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
# --options jsonfile_prefix=${WORK_DIR}/base_results

# bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_novel_train.py \
# ${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
# --options jsonfile_prefix=${WORK_DIR}/novel_results

# bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
# ${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
# --options jsonfile_prefix=${WORK_DIR}/all_results
