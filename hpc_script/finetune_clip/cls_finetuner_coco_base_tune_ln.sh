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

# finetune the clip with all COCO base categories, finetuning the LN only

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
rm -rf ./data
ln -sf /project/nevatia_174/zhuoming/detection ./data

PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/train.py \
    configs/cls_finetuner/cls_finetuner_coco_base_tune_ln.py --launcher pytorch \
    --work-dir=/project/nevatia_174/zhuoming/detection/test/cls_finetuner_coco_base_all_tune_ln
    #--resume-from=/project/nevatia_174/zhuoming/detection/test/new_rpn_patches246_coco/latest.pth