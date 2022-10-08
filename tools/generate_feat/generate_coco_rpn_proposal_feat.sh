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

cd /project/nevatia_174/zhuoming/code/new_rpn/mmdetection
#rm -rf ./data
#ln -sf /project/nevatia_174/zhuoming/detection ./data

## this script aim to generate feat for the LVIS dataset with the clip proposal generated for LVIS
## due to the long-tail property of the LVIS dataset, we can expect that we should not filter the base cate in the dataset
## since the novel categories only exist in about 300 images, and the whole dataset have nearly 100,000 images
## this ratio make filtering the base categories does not make sense.

# spliting the generation into other section to accelerate the procedure
bash tools/dist_test.sh \
configs/cls_finetuner/cls_finetuner_clip_full_coco.py \
data/exp_res/cls_finetuner_clip_base48_all_train/epoch_12.pth 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=data/coco/rpn_proposal_feat/base_finetuned/extract_feat \
--cfg-options data.test.ann_file=data/coco/annotations/instances_train2017.json  data.test.img_prefix=data/coco/train2017/ \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=data/coco/rpn_proposal_feat/base_finetuned \
model.test_cfg.use_pregenerated_proposal=data/coco/rpn_proposal/mask_rcnn_r50_fpn_2x_coco_2gpu_base48_reg_class_agno \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True