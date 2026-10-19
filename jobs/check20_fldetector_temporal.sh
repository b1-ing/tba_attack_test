#!/bin/bash --login
#SBATCH --job-name=ffl_check20
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-3:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check20_%j.out
#SBATCH --exclude=node871,node877

# Check 20: FLDetector temporal distance measurement.
#
# Runs 35 rounds of FedAvg FL with a fixed set of adversarial (cov_64_pos
# trigger) and benign clients. Records per-client weight update vectors
# each round and computes temporal distance ||Δw^t - Δw^{t-1}||₂ as a
# proxy for FLDetector's malicious_score.
#
# Prediction: fixed-trigger backdoor clients produce consistent updates
# round-over-round → temporal distance ≈ benign clients → FLDetector blind.
#
# 35 rounds × 10 clients × 2 poison rates = 700 client-round measurements.
# ~2-2.5h.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check20
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check20_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check20_fldetector_temporal.py" \
    --code-dir     "$WORK" \
    --warmup-dir   "$WARMUP_DIR" \
    --out          "$OUT" \
    --adv-clients  34 36 47 60 \
    --ben-clients  0 1 2 3 5 10 \
    --n-rounds     35 \
    --window       10 \
    --poison-rates 0.50 0.10 \
    --amplitude    0.50 \
    --n-test       500 \
    2>&1 | tee "$OUT/check20_fldetector_temporal.log"

echo "Done: $(date)"
