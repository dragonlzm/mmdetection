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

# 80 cates
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco \
#    --cfg-options optimizer.lr=0.0025
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/latest.pth

# 48 cates
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48 \
#    --cfg-options optimizer.lr=0.0025
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_base48/latest.pth

# 48 cates without centerness
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48 \
   --cfg-options optimizer.lr=0.0025
   #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# test the model 
bash tools/dist_test.sh configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48.py \
data/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/epoch_12.pth 2 --eval bbox \
--eval-options jsonfile_prefix=data/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/base \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.test_cfg.score_thr=0.0 model.test_cfg.max_per_img=300