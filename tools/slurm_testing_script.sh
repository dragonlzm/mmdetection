#!/bin/bash
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=1
#SBATCH --cpu-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --account=<account_id>

module purge
module load gcc/8.3.0
module load cuda/10.1.243
./program