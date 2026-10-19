#!/bin/bash --login
#SBATCH --job-name=ffl_check22
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-4:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check22_%j.out
#SBATCH --exclude=node871,node877

# Check 22: FLAME defense against cov_64_pos backdoor.
#
# Sweeps noise multiplier λ ∈ {0.000, 0.001, 0.010, 0.100} plus FedAvg
# baseline. Each condition runs 25 rounds of FL (10 clients: 4 ADV + 6 BEN).
#
# Key questions:
#   1. Does HDBSCAN clustering exclude ADV clients? (λ=0 condition)
#   2. At what λ does noise kill the backdoor? (λ sweep)
#   3. What clean-accuracy cost does that λ impose?
#
# 5 conditions × 25 rounds × 10 clients = 1250 measurements. ~3h.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check22
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check22_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check22_flame.py" \
    --code-dir     "$WORK" \
    --warmup-dir   "$WARMUP_DIR" \
    --out          "$OUT" \
    --adv-clients  34 36 47 60 \
    --ben-clients  0 1 2 3 5 10 \
    --n-rounds     25 \
    --lambda-vals  0.000 0.001 0.010 0.100 \
    --poison-rate  0.50 \
    --amplitude    0.50 \
    --n-test       500 \
    2>&1 | tee "$OUT/check22_flame.log"

echo "Done: $(date)"
