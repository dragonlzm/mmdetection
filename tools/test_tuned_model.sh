#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100:2
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

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/epoch_12.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/results \
     --cfg-options model.test_cfg.crop_size_modi=1

python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/epoch_12.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/results \
     --cfg-options model.test_cfg.crop_size_modi=0.9

python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/epoch_12.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/results \
     --cfg-options model.test_cfg.crop_size_modi=0.75

python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/epoch_12.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_full_coco/results \
     --cfg-options model.test_cfg.crop_size_modi=0.5