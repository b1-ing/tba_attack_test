#!/bin/bash --login
#SBATCH --job-name=ffl_grid
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH --array=0-1
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_grid_%A_%a.out

# Poison-rate x poison-epochs grid on frozen checkpoints from jobs/train_ckpt.sh.
# Array task 0 = epoch-100 checkpoint (rho 0.5), task 1 = epoch-1050 checkpoint (rho 0.3).
# Results (grid.csv, probes.npz, prediction.txt) go to $HOME/ffl_grid/e<epoch>; analyse with jobs/grid_analyse.py.
# The script hard-fails (non-zero exit) if any cell's filter decision differs from the reference's.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

EPOCHS_LIST=(100 1050)
CK=${EPOCHS_LIST[$SLURM_ARRAY_TASK_ID]}
CKPT=$HOME/ffl_ckpt/ckpt_e${CK}.pt
OUT=$HOME/ffl_grid/e${CK}

if [ ! -f "$PROJ/FilterFL/main.py" ]; then
    echo "ERROR: $PROJ/FilterFL/main.py missing - run 'git submodule update --init' in $PROJ" >&2
    exit 1
fi
[ -f "$CKPT" ] || { echo "ERROR: $CKPT missing - run jobs/train_ckpt.sh first" >&2; exit 1; }

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_grid_${SLURM_ARRAY_JOB_ID}_e${CK}
mkdir -p "$WORK" "$OUT"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

nvidia-smi -L || echo "WARNING: no GPU visible"
python "$PROJ/jobs/poison_grid.py" --code-dir "$WORK" --ckpt "$CKPT" --outdir "$OUT" \
    --rates 0.10 0.20 0.25 0.30 0.35 0.50 --poison-epochs 2 6 10 --reps 3

echo "Done: $(date)"
