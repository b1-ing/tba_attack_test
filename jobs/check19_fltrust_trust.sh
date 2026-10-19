#!/bin/bash --login
#SBATCH --job-name=ffl_check19
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check19_%j.out
#SBATCH --exclude=node871,node877

# Check 19: FLTrust trust score measurement.
#
# Measures TS_i = ReLU(cos(g_client, g_server)) for 7 trigger types at
# two poison rates (0.50 and 0.10), alongside FilterFL feat_align and s3.
#
# Core question: is there ANY trigger that simultaneously evades FilterFL
# (s3 < 0.10) AND survives FLTrust aggregation (trust_score > 0.50)?
# If not, the two defenses cover each other's blind spots — which is the
# paper's structural argument.
#
# 7 triggers × 2 poison_rates × 4 clients × 3 seeds = 168 measurements.
# ~1.5-2h (includes server root-dataset training once per run).

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check19
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check19_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check19_fltrust_trust.py" \
    --code-dir       "$WORK" \
    --warmup-dir     "$WARMUP_DIR" \
    --out            "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --poison-rates 0.50 0.10 \
    --root-per-class 10 \
    --amplitude 0.50 \
    --n-test 500 \
    2>&1 | tee "$OUT/check19_fltrust_trust.log"

echo "Done: $(date)"
