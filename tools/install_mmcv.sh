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

cd /project/nevatia_174/zhuoming/code
cd mmcv

pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install -e .

MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e .


salloc --partition=gpu --gres=gpu:v100:1 --time=2:00:00 --cpus-per-task=1 --mem=16GB --account=nevatia_174