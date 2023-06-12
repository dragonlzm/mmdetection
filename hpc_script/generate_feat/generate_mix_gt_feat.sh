#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --account=nevatia_174

# extract the feature using multiple size of the bbxoes for COCO zeroshot setting

module purge
module load gcc/8.3.0
#module load cuda/10.1.243
#./program

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

# for the finetuned clip generated proposal(filter base cates)
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
   /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
   configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
   data/exp_res/cls_finetuner_clip_base48_all_train/latest.pth \
   --launcher pytorch --eval=gt_acc \
   --eval-options jsonfile_prefix=data/coco/clip_proposal_feat/base48_finetuned_base_filtered/mix_gt \
   --cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json data.test.img_prefix=data/coco/train2017/ \
   model.test_cfg.generate_mix_gt_feat=True model.test_cfg.feat_save_path=data/coco/clip_proposal_feat/base48_finetuned_base_filtered