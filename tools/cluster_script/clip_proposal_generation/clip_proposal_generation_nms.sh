#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=24:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data

# 16_32_512 nms all anchor
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_16_32_512_nms_on_all_fixed \
#    --cfg-options model.anchor_generator.strides=[16] model.test_cfg.nms_on_all_anchors=True

# 16_32_512 nms all anchor nms07
#PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_16_32_512_nms_on_all_07_fixed \
#    --cfg-options model.anchor_generator.strides=[16] model.test_cfg.nms_on_all_anchors=True model.test_cfg.nms_threshold=0.7

# 16_32_512 nms all anchor
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_32_64_1024_nms_on_all_fixed \
    --cfg-options model.test_cfg.nms_on_all_anchors=True

# 16_32_512 nms all anchor nms07
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_32_64_1024_nms_on_all_07_fixed \
    --cfg-options model.test_cfg.nms_on_all_anchors=True model.test_cfg.nms_threshold=0.7