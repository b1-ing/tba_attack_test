#!/bin/bash --login
#SBATCH --job-name=ffl_check9
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check9_%j.out

# Check 9: Benign client with synthetically boosted class-0 share → Stage-3 spike?
# Direct mechanism test: if a benign client's Stage-3 score rises when its training
# data is enriched with class-0 samples (no backdoor injection), that confirms
# Stage-3 is measuring label-distribution conflation, not backdoor activity.
#
# Sweeps class-0 fraction in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50] for client 1
# (natural share=0.000), 3 training seeds each = 18 measurements.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check9
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check9_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check9_share_inject.py" \
    --code-dir "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out "$OUT" \
    --client 1 \
    --fracs 0.0 0.05 0.10 0.20 0.30 0.50 \
    --seeds 1 2 3 \
    2>&1 | tee "$OUT/check9_share_inject.log"

echo "Done: $(date)"
