#!/bin/bash --login
#SBATCH --job-name=ffl_check21
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-4:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check21_%j.out
#SBATCH --exclude=node871,node877

# Check 21: Foolsgold defense vs same-trigger and diverse-trigger attack.
#
# Three conditions, 35 rounds each:
#   (A) FedAvg + same trigger (cov_64_pos, shared mask) — baseline
#   (B) Foolsgold + same trigger — prediction: ADV weights → 0, backdoor fails
#   (C) Foolsgold + diverse trigger — each ADV client has unique 64px mask;
#       prediction: updates decorrelated, Foolsgold weights recover, backdoor survives
#
# 3 conditions × 35 rounds × 10 clients = 1050 client-round measurements.
# ~3-3.5h.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check21
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check21_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check21_foolsgold.py" \
    --code-dir     "$WORK" \
    --warmup-dir   "$WARMUP_DIR" \
    --out          "$OUT" \
    --adv-clients  34 36 47 60 \
    --ben-clients  0 1 2 3 5 10 \
    --n-rounds     35 \
    --poison-rate  0.50 \
    --amplitude    0.50 \
    --n-test       500 \
    2>&1 | tee "$OUT/check21_foolsgold.log"

echo "Done: $(date)"
