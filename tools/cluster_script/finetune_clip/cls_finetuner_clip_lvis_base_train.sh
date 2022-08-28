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

WORK_DIR="data/exp_res/cls_finetuner_clip_lvis_base_train"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_base_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/base_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_lvis_novel_train.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/novel_results

bash tools/dist_test.sh configs/cls_finetuner/cls_finetuner_clip_full_lvis.py \
${WORK_DIR}/epoch_12.pth 2 --eval=gt_acc \
--options jsonfile_prefix=${WORK_DIR}/all_results
