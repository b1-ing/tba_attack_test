#!/bin/bash --login
#SBATCH --job-name=ffl_check3b
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check3b_%j.out

# Check 3b: Extended UAP correlation (tasks 1 & 2)
# Adds 20 unflagged benign clients spanning the class-0 share range.
# Reports within-group Spearman and causal chain (share → UAP → Stage-3).

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
PROBE_NPZ=$HOME/ffl_probe/s1_ours/probes.npz
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check3b
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check3b_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check3b_uap_extended.py" \
    --code-dir "$WORK" \
    --out "$OUT" \
    --probe-npz "$PROBE_NPZ" \
    --warmup-dir "$WARMUP_DIR" \
    --skip-warmup \
    --rates 0.3 0.5 \
    --adv-clients 34 36 45 47 \
    --n-flagged 5 \
    --n-unflagged 20 \
    --uap-eps 0.1 \
    --uap-steps 300 \
    2>&1 | tee "$OUT/check3b_uap.log"

echo "Done: $(date)"
