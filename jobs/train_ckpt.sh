#!/bin/bash --login
#SBATCH --job-name=ffl_ckpt
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 4-0:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_ckpt_%j.out

# Unmodified reference FilterFL, CIFAR10, --defense ours, saving global-model checkpoints at epochs 100 and 1050
# (1050 is unambiguously inside the rho=0.3 regime; 100 is rho=0.5). Checkpoints go to $HOME (scratch is auto-cleaned).
# last.pt is refreshed every 50 epochs; if the job hits the time limit, resubmit with
#   sbatch --export=ALL,RESUME=1 jobs/train_ckpt.sh

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
CKPT_DIR=$HOME/ffl_ckpt
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

if [ ! -f "$PROJ/FilterFL/main.py" ]; then
    echo "ERROR: $PROJ/FilterFL/main.py missing - run 'git submodule update --init' in $PROJ" >&2
    exit 1
fi

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_ckpt_${SLURM_JOB_ID}
mkdir -p "$WORK" "$CKPT_DIR"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

nvidia-smi -L || echo "WARNING: no GPU visible"
RESUME_FLAG=""
if [ -n "${RESUME:-}" ]; then RESUME_FLAG="--resume"; fi

python "$PROJ/jobs/train_ckpt.py" --code-dir "$WORK" --ckpt-dir "$CKPT_DIR" \
    --epochs 1050 --save-at 100 1050 --save-every 50 $RESUME_FLAG

ls -l "$CKPT_DIR"
echo "Done: $(date)"
