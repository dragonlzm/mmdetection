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

# mask rcnn 1x albu augmentation
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_my_albu.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_my_albu
   #--resume-from=/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_my_albu/latest.pth


# for 2*2 novel17 (should be delete in this script)
TRAIN_CONFIG="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17.py"
WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/mask_rcnn_r50_fpn_1x_coco_detectron_2x2_novel17_reg_class_agno"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    ${TRAIN_CONFIG} --launcher pytorch \
    --work-dir=${WORK_DIR} \
    --cfg-options model.roi_head.bbox_head.reg_class_agnostic=True \
    --resume-from=${WORK_DIR}/latest.pth  