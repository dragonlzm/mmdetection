#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=30:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# 48 cates 100 clip proposal
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_w01 \
#    --cfg-options model.train_cfg.distill_loss_factor=0.1
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_w002 \
#    --cfg-options model.train_cfg.distill_loss_factor=0.02
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_w05 \
#    --cfg-options model.train_cfg.distill_loss_factor=0.5
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# 48 cates 200 clip proposal
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
   configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro.py --launcher pytorch \
   --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01 \
   --cfg-options model.train_cfg.distill_loss_factor=0.1
   #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w002 \
#    --cfg-options model.train_cfg.distill_loss_factor=0.02
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#    configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro.py --launcher pytorch \
#    --work-dir=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w05 \
#    --cfg-options model.train_cfg.distill_loss_factor=0.5
#    #--resume-from=/project/nevatia_174/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco_wo_centerness_base48/latest.pth

# test the model 
bash tools/dist_test.sh configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_clip_feat_1x_coco_base48.py \
data/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/epoch_12.pth 2 --eval bbox \
--eval-options jsonfile_prefix=data/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/base \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.test_cfg.score_thr=0.0 model.test_cfg.max_per_img=300

bash tools/dist_test.sh configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48.py \
/data/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/epoch_12.pth 2 --eval bbox \
--eval-options jsonfile_prefix=/data/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/base_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.test_cfg.score_thr=0.0 model.test_cfg.max_per_img=300

bash tools/dist_test.sh configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_novel17.py \
/data/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/epoch_12.pth 2 --eval bbox \
--eval-options jsonfile_prefix=/data/zhuoming/detection/fcos/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_base48_200_clip_pro_w01/novel_results \
--cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
model.test_cfg.score_thr=0.0 model.test_cfg.max_per_img=300