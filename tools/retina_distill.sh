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

####### update the hyper parameters
bash tools/new_dist_train_23.sh configs/retinanet_distill/retinanet_distill_r50_fpn_1x_coco_base48.py 2 \
data/one_stage/retina_distill_ori_1 /data/zhuoming/detection \
--cfg-options optimizer.lr=0.0025 \
--resume-from=data/one_stage/retina_distill_ori_1/epoch_3.pth



# # test the model 
# bash tools/dist_test.sh configs/fcos_distill/fcos_r50_caffe_fpn_gn-head_with_base48_tuned_clip_feat_1x_coco_bn65.py \
# ${WORK_DIR}/latest.pth 2 --eval bbox \
# --eval-options jsonfile_prefix=${WORK_DIR}/base_and_novel \
# --cfg-options data.test.eval_filter_empty_gt=False data.test.ann_file=data/coco/annotations/instances_val2017_65cates.json \
# model.roi_head.bbox_head.reg_with_cls_embedding=True data.test.eval_on_splits='zeroshot'