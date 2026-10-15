#!/bin/bash --login
#SBATCH --job-name=ffl2_cifar
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 2-0:00
#SBATCH --array=0-1
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl2_cifar_%A_%a.out

# Unmodified reference FilterFL (FilterFL/), CIFAR10: no defence vs defence "ours".
# Array task 0 = no defence, task 1 = ours. Both use cifar_params.yaml as-is
# (seed 1, 2000 epochs) unless EPOCHS is set, e.g.  sbatch --export=ALL,EPOCHS=50 jobs/cifar_nodef_vs_ours.sh
# The reference code is copied into a per-task scratch dir so nothing in the repo is written to.
# NOTE: the partition (-p) / GPU request (-G) above are a guess; adjust to the cluster's GPU queue.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch

DEFENCES=(None ours)
DEFENCE=${DEFENCES[$SLURM_ARRAY_TASK_ID]}

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_cifar_${SLURM_ARRAY_JOB_ID}_${DEFENCE}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models   # handle.py does os.mkdir without parents

# Share the CIFAR10 download between the two tasks if it already exists.
if [ -d "$SCRATCH/ffl_data" ]; then ln -s "$SCRATCH/ffl_data" data; fi

PARAMS=cifar_params.yaml
if [ -n "${EPOCHS:-}" ]; then
    sed "s/^epochs: .*/epochs: $EPOCHS/" cifar_params.yaml > cifar_params_run.yaml
    PARAMS=cifar_params_run.yaml
fi

echo "Host: $(hostname) | defence=$DEFENCE | params=$PARAMS | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible (device.py needs cuda:0)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

python main.py --params "$PARAMS" --defense "$DEFENCE"

echo "--- final results (defence=$DEFENCE) ---"
LOG=$(ls -d saved_models/model_*/ | head -n1)log.txt
grep "Global_Test_Poison:False" "$LOG" | tail -n1   # clean accuracy
grep "Global_Test_Poison:True"  "$LOG" | tail -n1   # attack success rate
grep "count_all_attack" "$LOG" | tail -n1 || true   # defence filter stats (ours only)
echo "Log: $WORK/$LOG"
echo "Done: $(date)"
