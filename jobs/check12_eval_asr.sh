#!/bin/bash --login
#SBATCH --job-name=ffl_check12e
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 2
#SBATCH -t 0-0:15
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check12e_%j.out
#SBATCH --exclude=node871

# Post-run ASR evaluation for check12 (checker trigger).
# Loads the final global model saved by main.py, applies the checker trigger
# to the test set, reports true ASR. Run this after check12 (23260868) completes.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

# The checkpoint saved by main.py — find it in the check12 scratch dir
# main.py saves to saved_models/global_model_*.pt or similar
CKPT_DIR=$SCRATCH/ffl_check12_23260868/saved_models

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check12e_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

# Find the latest checkpoint in the scratch saved_models dir
CKPT=$(ls -t "$CKPT_DIR"/*.pt "$CKPT_DIR"/*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found in $CKPT_DIR"
    ls "$CKPT_DIR" || true
    exit 1
fi
echo "Using checkpoint: $CKPT"

python "$PROJ/jobs/check12_eval_asr.py" \
    --code-dir  "$WORK" \
    --checkpoint "$CKPT" \
    --amplitude 0.50 \
    2>&1 | tee "$HOME/ffl_check12/check12_eval_asr.log"

echo "Done: $(date)"
