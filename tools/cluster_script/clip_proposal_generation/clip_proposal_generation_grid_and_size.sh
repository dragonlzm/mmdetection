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

cd /home1/liuzhuom/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data

# 32_64_1024
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_32_64_1024

# 16_32_512
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_16_32_512 \
    --cfg-options model.anchor_generator.strides=[16]

# 16_16_1024
PYTHONPATH="/home1/liuzhuom/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /home1/liuzhuom/mmdetection/tools/test.py \
    /home1/liuzhuom/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=/project/nevatia_174/zhuoming/detection/test/cls_proposal_generator_coco/results_16_16_1024 \
    --cfg-options model.anchor_generator.strides=[16] model.anchor_generator.scales=[2, 4, 8, 16, 32]
