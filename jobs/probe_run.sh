#!/bin/bash --login
#SBATCH --job-name=ffl_probe
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_probe_%A_%a.out

# Reference FilterFL, CIFAR10, alpha=0.5, poison_rate=0.3 (the paper's stated defaults), with per-round probe responses logged
# by jobs/probe_run.py (reference code unmodified). Array tasks:
#   0 = seed 2, no defence   1 = seed 2, ours   2 = seed 1, ours (re-run of 23200242 task 1 with probes; should match its counts)
# Output: $HOME/ffl_probe/s<seed>_<defence>/probes.npz

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

SEEDS=(2 2 1)
DEFS=(None ours ours)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DEF=${DEFS[$SLURM_ARRAY_TASK_ID]}
OUT=$HOME/ffl_probe/s${SEED}_${DEF}

if [ ! -f "$PROJ/FilterFL/main.py" ]; then
    echo "ERROR: $PROJ/FilterFL/main.py missing - run 'git submodule update --init' in $PROJ" >&2
    exit 1
fi

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_probe_${SLURM_ARRAY_JOB_ID}_s${SEED}_${DEF}
mkdir -p "$WORK" "$OUT"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | seed=$SEED defence=$DEF | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"
python "$PROJ/jobs/probe_run.py" --code-dir "$WORK" --outdir "$OUT" --defense "$DEF" \
    --seed "$SEED" --alpha 0.5 --poison-rate 0.3

grep "count_all_attack" saved_models/model_*/log.txt | tail -n1 || true
echo "Done: $(date)"
