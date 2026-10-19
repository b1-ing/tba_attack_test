#!/bin/bash --login
#SBATCH --job-name=ffl_check11s
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 2
#SBATCH -t 0-0:20
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check11s_%j.out
#SBATCH --exclude=node871

# Check 11 (spectral): Verify that the GAN generator's output is concentrated
# at low spatial frequencies while the checkerboard trigger sits at Nyquist.
# No training — just one GAN forward pass + FFT analysis.

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

WORK=$SCRATCH/ffl_check11s_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check11_spectral.py" \
    --code-dir   "$WORK" \
    --warmup-dir "$WARMUP_DIR" \
    --out        "$OUT" \
    --n-gan      200 \
    --amplitude  0.50 \
    2>&1 | tee "$OUT/check11_spectral.log"

echo "Done: $(date)"
