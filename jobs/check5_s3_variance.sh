#!/bin/bash --login
#SBATCH --job-name=ffl_check5
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check5_%j.out

# Check 5: Stage-3 variance decomposition.
# Separates GAN probe noise from attacker training noise for clients 45 and 47.
# Full factorial: 5 attacker seeds × 5 GAN seeds = 25 s3 measurements per client.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

CLIENTS=${CLIENTS:-"45 47"}
RATE=${RATE:-0.5}
ATTACKER_SEEDS=${ATTACKER_SEEDS:-"1 2 3 4 5"}
GAN_SEEDS=${GAN_SEEDS:-"1 2 3 4 5"}

OUT=$HOME/ffl_check5
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check5_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | clients=$CLIENTS | rate=$RATE | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check5_s3_variance.py" \
    --code-dir "$WORK" \
    --out "$OUT" \
    --warmup-dir "$WARMUP_DIR" \
    --skip-warmup \
    --clients $CLIENTS \
    --rate "$RATE" \
    --attacker-seeds $ATTACKER_SEEDS \
    --gan-seeds $GAN_SEEDS \
    2>&1 | tee "$OUT/check5_s3_variance.log"

echo "Done: $(date)"
