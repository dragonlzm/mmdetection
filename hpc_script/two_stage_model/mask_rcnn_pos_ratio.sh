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

cd /project/nevatia_174/zhuoming/code/pos_ratio/mmdetection
# rm -rf ./data
# ln -sf /project/nevatia_174/zhuoming/detection ./data


# for 2*2
TRAIN_CONFIG="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_detectron_2x2.py"
WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_ratio_exp"
PYTHONPATH="/project/nevatia_174/zhuoming/code/pos_ratio/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/pos_ratio/mmdetection/tools/train.py \
    ${TRAIN_CONFIG} --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth    


# for 2*2 base48
# TRAIN_CONFIG="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_base48.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_base48_ratio_exp"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/pos_ratio/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/pos_ratio/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth   

