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

# for faster rcnn baseline training

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data


# training with all categories
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_2gpu.py --launcher pytorch \
#     --work-dir=/project/nevatia_174/zhuoming/detection/test/faster_rcnn_r50_fpn_1x_coco_2gpu 
#     #--resume-from=/project/nevatia_174/zhuoming/detection/test/new_rpn_patches246_coco/latest.pth

# faster rcnn base48
TRAIN_CONFIG="configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco_base48.py"
BASE_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base48.py"
NOVEL_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel17.py"
WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/faster_rcnn_r50_caffe_c4_1x_coco_base48"
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    ${TRAIN_CONFIG} --launcher pytorch \
    --work-dir=${WORK_DIR} \
    #--resume-from=${WORK_DIR}/latest.pth

# faster rcnn base60
# TRAIN_CONFIG="configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco_base60.py"
# BASE_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base60.py"
# NOVEL_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel20.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/faster_rcnn_r50_caffe_c4_1x_coco_base60"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# faster rcnn base48
# TRAIN_CONFIG="configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco_with_pretrain_base48.py"
# BASE_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base48.py"
# NOVEL_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel17.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/faster_rcnn_r50_caffe_c4_1x_coco_with_pretrain_base48"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# faster rcnn base60
# TRAIN_CONFIG="configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco_with_pretrain_base60.py"
# BASE_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_base60.py"
# NOVEL_TEST_CONFIG="configs/rpn/rpn_r50_caffe_c4_1x_coco_novel20.py"
# WORK_DIR="/project/nevatia_174/zhuoming/detection/baseline/faster_rcnn_r50_caffe_c4_1x_coco_with_pretrain_base60"
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
#     ${TRAIN_CONFIG} --launcher pytorch \
#     --work-dir=${WORK_DIR} \
#     #--resume-from=${WORK_DIR}/latest.pth

# test the model
CHECKPOINT_NAME="epoch_12.pth"
#CHECKPOINT_NAME="epoch_24.pth"
ALL_CONFIG="configs/rpn/rpn_r50_fpn_1x_coco.py"

bash tools/dist_test.sh ${ALL_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_all

bash tools/dist_test.sh ${BASE_TEST_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_base

bash tools/dist_test.sh ${NOVEL_TEST_CONFIG} \
${WORK_DIR}/${CHECKPOINT_NAME} 2 \
--eval=proposal_fast --options jsonfile_prefix=${WORK_DIR}/test_novel
