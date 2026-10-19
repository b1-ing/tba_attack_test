#!/bin/bash --login
#SBATCH --job-name=ffl_check17
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check17_%j.out
#SBATCH --exclude=node871,node877

# Check 17: Effective rank analysis.
#
# Re-trains local models for each trigger from check16 and measures three
# quantities that together explain the cov_1024 anomaly (detected at full
# coverage because all-positive sign creates a uniform brightness direction
# that GAN probes can activate):
#
#   delta_W_eff_rank       — effective rank of ΔW_fc (SVD participation ratio)
#   spatial_entropy        — entropy of per-pixel gradient magnitude map
#   probe_detectable_mass  — fraction of ΔW_fc energy aligned with probe centroid
#
# 9 trigger types × 4 clients × 3 seeds = 108 measurements. ~1.5-2h.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check17
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check17_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check17_rank_analysis.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --amplitude 0.50 \
    --n-ref 50 \
    --cos-thresh 0.10 \
    2>&1 | tee "$OUT/check17_rank_analysis.log"

echo "Done: $(date)"
