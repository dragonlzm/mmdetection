#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=1:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    configs/mask_rcnn_distill/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_novel17.py \
#    /project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_base48_tuned_r50_fpn_1x_coco_2gpu_base48/epoch_12.pth \
#    --launcher pytorch --eval bbox \
#    --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_base48_tuned_r50_fpn_1x_coco_2gpu_base48/test_results

#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    configs/mask_rcnn_distill/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_novel17.py \
#    /project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_base48_distill_fixed/epoch_12.pth \
#    --launcher pytorch --eval bbox \
#    --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_base48_distill_fixed/test_results

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    configs/mask_rcnn_distill/mask_rcnn_distill_r50_fpn_1x_coco_2gpu_novel17.py \
    /project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_base48_tuned_r50_fpn_2x_coco_2gpu_base48/latest.pth \
    --launcher pytorch --eval bbox \
    --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/exp_res/mask_rcnn_distill_base48_tuned_r50_fpn_2x_coco_2gpu_base48/test_results