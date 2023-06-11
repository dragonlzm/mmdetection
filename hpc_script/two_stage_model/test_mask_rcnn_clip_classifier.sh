#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=6:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_clip_classifier_base48.py \
# data/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/epoch_12.pth 2 \
# --eval bbox segm --options jsonfile_prefix=data/mask_rcnn_clip_classifier/results_base48 

bash tools/dist_test.sh configs/mask_rcnn_distill/mask_rcnn_clip_classifier_novel17.py \
data/grad_clip_check/mask_rcnn_with_base48_tuned_clip_feat_r50_fpn_1x_coco_base48_gn_10_200clipproposal/epoch_12.pth 2 \
--eval bbox segm --options jsonfile_prefix=data/mask_rcnn_clip_classifier/results_novel17
