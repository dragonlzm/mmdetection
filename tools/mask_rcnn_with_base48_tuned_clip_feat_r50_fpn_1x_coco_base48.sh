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

# for 1x training
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48
    #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_base48/latest.pth


# for 1x training
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/mask_rcnn_distill/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_updated
    #--resume-from=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_base48/latest.pth