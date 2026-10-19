#!/bin/bash --login
#SBATCH --job-name=ffl_check11
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check11_%j.out
#SBATCH --exclude=node871

# Check 11: Can trigger design evade Stage-3?
#
# For each (client in {34, 36}, trigger_type in {patch, checker, noise}, seed in {1,2,3}):
#   - 1 poisoned round from warmup checkpoint at rate=0.50
#   - Measure: feat_align (Stage-3 proxy), trigger_feat_al (where trigger lives in
#     feature space), evasion_gap = trigger_feat_al - feat_align, s3, asr_proxy
#
# The evasion gap is the key metric: if the trigger lives in a direction that aligns
# with W_fc[target] (high trigger_feat_al) but the GAN's probe images don't find it
# (low feat_align), then Stage-3 misses the backdoor.
#
# 18 measurements per client = 36 total. Runtime: ~20–30 min.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check11
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check11_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check11_trigger_design.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 \
    --triggers patch checker noise \
    --rate 0.50 \
    --amplitude 0.50 \
    --seeds 1 2 3 \
    --n-test 500 \
    2>&1 | tee "$OUT/check11_trigger_design.log"

echo "Done: $(date)"
