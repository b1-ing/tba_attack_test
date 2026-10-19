#!/bin/bash --login
#SBATCH --job-name=ffl_check15
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check15_%j.out
#SBATCH --exclude=node871

# Check 15: Swap probe generator — GAN vs real held-out server data.
#
# Tests whether global triggers evade because:
#   H_GAN: the GAN's natural-image prior is the specific problem
#   H_NAT: any natural-image probe fails (structural blindspot)
#
# Measures s3 and feat_align under 3 probe regimes for the same 4 triggers
# as check14: gan / real_mixed (all classes) / real_cls0 (target class only).
# Also computes direct manifold distance: cos(trigger_features, probe_centroid)
# through the global model.
#
# 4 triggers × 4 clients × 3 seeds × 3 probe types = 144 measurements.
# Runtime: ~45-75 min.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1

OUT=$HOME/ffl_check15
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check15_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check15_probe_swap.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --clients 34 36 47 60 \
    --seeds 1 2 3 \
    --n-test 500 \
    --n-probe 200 \
    2>&1 | tee "$OUT/check15_probe_swap.log"

echo "Done: $(date)"
