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

## this is for training the RPN base line

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data


# rpn base48
# TRAIN_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base48.py"
# TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel17.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_caffe_c4_1x_coco_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# rpn base60
# TRAIN_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base60.py"
# TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel20.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_caffe_c4_1x_coco_base60"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth


# rpn base48 fpn
ALL_CONFIG="configs/rpn/rpn_r50_fpn_1x_coco.py"
TRAIN_CONFIG="configs/rpn/rpn_r50_fpn_1x_coco_base48.py"
TEST_CONFIG="configs/rpn/rpn_r50_fpn_1x_coco_novel17.py"
WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/rpn_r50_fpn_1x_coco_base48"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    ${TRAIN_CONFIG} --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth


# test the model
CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"

bash tools/dist_test.sh ${ALL_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_all

bash tools/dist_test.sh ${TRAIN_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_48

bash tools/dist_test.sh ${TEST_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_17


