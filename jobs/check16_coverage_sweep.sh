#!/bin/bash --login
#SBATCH --job-name=ffl_check16
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check16_%j.out
#SBATCH --exclude=node871

# Check 16: Coverage threshold sweep.
#
# Sweeps trigger spatial coverage from 16 to 1024 pixels (7 levels) to find
# the transition point where Stage-3 detection fails. Also computes the
# evasion_score (cos(trigger_feat, GAN_centroid) through global model) for
# each level, to verify it is monotone with coverage.
#
# Triggers: additive, random pixel positions, amplitude=0.5.
# References: patch (16px replacement), checker (1024px additive).
#
# 9 trigger types × 4 clients × 3 seeds = 108 measurements. Runtime: ~1.5-2h.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check16
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check16_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check16_coverage_sweep.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --amplitude 0.50 \
    --n-test 500 \
    2>&1 | tee "$OUT/check16_coverage_sweep.log"

echo "Done: $(date)"
