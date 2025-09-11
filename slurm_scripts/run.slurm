#!/bin/bash
#SBATCH --job-name=vp
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -D /data/gpfs/projects/punim1943/xunli/mr_benchmark/mr_codes
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=ALL

set -Eeuo pipefail
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

# 激活你刚建的环境（用绝对路径）
conda activate /data/gpfs/projects/punim1943/conda/vps

# 保险：避免 ~/.local 污染（可留着）
export PYTHONNOUSERSITE=1
unset PYTHONPATH

mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi || true

srun --export=ALL --unbuffered python -u main.py

