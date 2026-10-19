#!/bin/bash --login
#SBATCH --job-name=ffl_check18
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check18_%j.out
#SBATCH --exclude=node871,node877

# Check 18: Sign pattern and gradient-guided position selection at k=64px.
#
# Isolates two degrees of freedom at fixed coverage=64px:
#   (1) Sign pattern: pos / neg / randsign
#   (2) Position selection: random / gradient-lowest / gradient-highest / grad+sign
#   (3) Version A feasibility: 20-step gradient optimisation of trigger shape
#
# 6 triggers × 4 clients × 3 seeds = 72 measurements. ~45-60 min.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check18
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check18_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check18_sign_and_grad.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --k 64 \
    --amplitude 0.50 \
    --opt-steps 20 \
    --opt-lr 0.01 \
    --n-test 500 \
    2>&1 | tee "$OUT/check18_sign_and_grad.log"

echo "Done: $(date)"
