#!/bin/bash --login
#SBATCH --job-name=ffl_check14
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check14_%j.out
#SBATCH --exclude=node871

# Check 14: Isolate the patch-vs-BLN confound.
#
# The patch (local pixel replacement) gives s3≈0.225; BLN (global additive) gives s3≈0.
# Three variables differ: modification type (replacement vs additive), spatial extent
# (16 px vs all pixels), and amplitude. This experiment holds two fixed, varies one:
#
#   patch             — local, replacement, amp≈0.7   [KNOWN: s3≈0.225]
#   additive_local_05 — local, additive,    amp=0.5   [varies TYPE]
#   additive_local_hi — local, additive,    amp=1.0   [varies TYPE+AMP]
#   replace_global    — global, replacement, amp≈0.7  [varies EXTENT]
#
# 4 triggers × 4 clients × 3 seeds = 48 measurements. Runtime: ~30–45 min.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check14
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check14_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check14_isolation.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --n-test 500 \
    2>&1 | tee "$OUT/check14_isolation.log"

echo "Done: $(date)"
