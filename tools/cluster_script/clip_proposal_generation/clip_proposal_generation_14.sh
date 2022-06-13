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

# 1
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_0_8000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 2
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_2 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_8000_16000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 3
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_3 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_16000_24000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 4
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_4 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_24000_32000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 5
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_5 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_32000_40000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 6
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_6 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_40000_48000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 7
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_7 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_48000_56000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 8
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_8 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_56000_64000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 9
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_9 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_64000_72000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 10
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_10 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_72000_80000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True


# 11
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_11 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_80000_88000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 12
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_88000_96000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 13
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_96000_104000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 14
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_12 \
    --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
    model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
    data.test.ann_file=data/coco/annotations/instances_train2017_104000_112000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True

# 15
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_112000_120000.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True


# the remain
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/epoch_12.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/bg_results_32_64_512_remain \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_bg_proposal/32_64_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017.json data.test.eval_filter_empty_gt=False model.test_cfg.least_conf_bbox=True