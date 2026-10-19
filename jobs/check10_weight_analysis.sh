#!/bin/bash --login
#SBATCH --job-name=ffl_check10
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check10_%j.out

# Check 10: What per-update property does Stage-3 read?
# Measures weight-space quantities alongside Stage-3 for clients 34/36/45/47
# at rates 0.10–0.50 × 3 seeds = 72 measurements.
# No checkpoints needed — retrains from warmup in ~30 minutes.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check10
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check10_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check10_weight_analysis.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 45 47 \
    --rates   0.10 0.20 0.25 0.30 0.40 0.50 \
    --seeds   1 2 3 \
    2>&1 | tee "$OUT/check10_weight_analysis.log"

echo "Done: $(date)"
