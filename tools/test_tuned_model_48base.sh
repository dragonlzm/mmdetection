#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=24:00:00
#SBATCH --account=nevatia_174


module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /home1/liuzhuom/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data

# base
#PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=2 \
#    /home1/liuzhuom/mmdetection/tools/test.py \
#    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
#    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_novel17_all_train.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results