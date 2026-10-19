#!/bin/bash --login
#SBATCH --job-name=ffl_check13
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check13_%j.out
#SBATCH --exclude=node871

# Check 13: Frequency–evasion trade-off curve.
#
# Sweeps band-limited triggers at median normalised frequencies 0.25–0.99,
# plus reference triggers (patch, checker, noise), measuring:
#   s3, feat_align, trigger_feat_al, evasion_gap, asr_proxy
# on the warmup checkpoint from check1 (~/ffl_check1/warmup_ckpt.pt).
#
# 16 BLN frequencies + 3 reference triggers = 19 trigger types
# × 4 clients × 3 seeds = 228 measurements. Runtime: ~1.5–2h on gpuL.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check13
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check13_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check13_freq_sweep.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --amplitude 0.50 \
    --bandwidth 0.10 \
    --n-test 500 \
    2>&1 | tee "$OUT/check13_freq_sweep.log"

echo "Done: $(date)"
