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
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_1 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_0_5000.json

# 2
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_2 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_5000_10000.json

# 3
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_3 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_10000_15000.json

# 4
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_4 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_15000_20000.json

# 5
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_5 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_20000_25000.json

# 6
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_6 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_25000_30000.json

# 7
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_7 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_30000_35000.json

# 8
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_8 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_35000_40000.json

# 9
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_9 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_40000_45000.json

# 10
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_10 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_45000_50000.json


# 11
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_11 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_50000_55000.json

# 12
# PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=2 \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
#     /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
#     /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
#     --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_12 \
#     --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
#     model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
#     data.test.ann_file=data/coco/annotations/instances_train2017_55000_60000.json

# the remain
PYTHONPATH="/project/nevatia_174/zhuoming/code/new_rpn/mmdetection":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/tools/test.py \
    /project/nevatia_174/zhuoming/code/new_rpn/mmdetection/configs/cls_proposal_generator/cls_proposal_generator_coco.py \
    /project/nevatia_174/zhuoming/detection/test/cls_finetuner_clip_base_all_train/latest.pth \
    --launcher pytorch --eval=proposal_fast --options jsonfile_prefix=data/test/cls_proposal_generator_coco/results_32_32_512_remain \
    --cfg-options model.anchor_generator.strides=[32] model.anchor_generator.scales=[1,2,4,8,16] model.test_cfg.nms_on_all_anchors=True \
    model.test_cfg.nms_threshold=0.7 model.test_cfg.min_entropy=True model.test_cfg.bbox_save_path_root=data/coco/clip_proposal/32_32_512 \
    data.test.ann_file=data/coco/annotations/instances_train2017.json data.test.eval_filter_empty_gt=False