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

# this script aims to test the classification accuracy of CLIP when attribute of each categories is provided
# On COCO dataset

cd /home1/liuzhuom/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data


PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco_multi_attr_test_single.py \
    /home1/liuzhuom/epoch_0.pth \
     --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results