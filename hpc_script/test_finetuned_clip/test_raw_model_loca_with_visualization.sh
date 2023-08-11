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

# This is testing the classification acc base on different perturbation on bbox location
# using raw CLIP, with visulization, On COCO dataset

cd /home1/liuzhuom/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data

# 0.2
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.2 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_novel17_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.2 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.2 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

# 0.25
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.25 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_novel17_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.25 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.25 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

# 0.33
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.33 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_novel17_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.33 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.33 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

# 0.5
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_base48_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.5 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_novel17_all_train.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.5 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model

PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
    /home1/liuzhuom/epoch_0.pth \
    --launcher pytorch --eval=gt_acc --options jsonfile_prefix=/home1/liuzhuom/mmdetection/results \
    --cfg-options model.test_cfg.crop_loca_modi=0.5 data.test.visualization_path=/home1/liuzhuom/mmdetection/visualization_loca_raw_model