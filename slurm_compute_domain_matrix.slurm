#!/bin/bash
#SBATCH --job-name=domain-matrix
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100
#SBATCH -D /data/gpfs/projects/punim1943/vp-run

set -Eeuo pipefail
module purge
module load foss/2022a
module load PyTorch/2.1.2-CUDA-12.2.0

mkdir -p logs results
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
nvidia-smi || true

ROOT=/data/gpfs/projects/punim1943/domainnet
OUT_PREFIX=results/domainnet

# 如需快速试跑，可改为 --max-batches 10 --kid-subsets 50 --sw-proj 64
srun --export=ALL --unbuffered \
  python -u compute_domain_matrix.py \
    --root "${ROOT}" \
    --device cuda \
    --out-prefix "${OUT_PREFIX}" \
    --sw-proj 256 --kid-subsets 200 --kid-size 100 --kl-diag

echo "Done. Matrices saved under ${OUT_PREFIX}_matrix_*.csv"
