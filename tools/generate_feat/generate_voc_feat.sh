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


# spliting the generation into other section to accelerate the procedure
#CHECKPOINT="data/exp_res/cls_finetuner_clip_lvis_base_train_over_sample_v2/epoch_18.pth"
CHECKPOINT="data/grad_clip_check/cls_finetuner_clip_voc_base15_split2_all_train/epoch_12.pth"
CONFIG_FILE="configs/cls_finetuner/cls_finetuner_clip_voc_bn20_all_train.py"
BBOX_SAVE_PATH_ROOT="data/VOCdevkit/clip_proposal/split1_32_32_512"
#FEAT_SAVE_PATH_ROOT="data/lvis_v1/clip_proposal_feat/lvis_base_finetuned"
FEAT_SAVE_PATH_ROOT="data/VOCdevkit/clip_proposal_feat/split1_base_finetuned"

#### update for using the best overall perf model to extract feature, no longer filter the base cate
bash tools/dist_test.sh \
${CONFIG_FILE} \
${CHECKPOINT} 2 \
--eval=gt_acc \
--eval-options jsonfile_prefix=${FEAT_SAVE_PATH_ROOT}/extract_feat \
--cfg-options data.test.ann_file=data/lvis_v1/annotations/lvis_v1_train.json \
model.test_cfg.generate_bbox_feat=True model.test_cfg.feat_save_path=${FEAT_SAVE_PATH_ROOT} \
model.test_cfg.use_pregenerated_proposal=${BBOX_SAVE_PATH_ROOT} \
model.test_cfg.num_of_rand_bboxes=500 model.test_cfg.save_cates_and_conf=True model.test_cfg.rand_select_subset=True
