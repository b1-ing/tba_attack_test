#!/bin/bash --login
#SBATCH --job-name=ffl_check7_ctrl
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check7_ctrl_%j.out

# Check 7 (Control): 2000-round FilterFL run at rate=0.5, alpha=0.5 with
# no evasion (bon_k=1, standard single-trial attack).
# Baseline showing what fraction of adversarial updates get filtered without
# any best-of-N selection.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

SEED=${SEED:-1}

OUT=$HOME/ffl_check7
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check7_ctrl_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | bon_k=1 (control) | seed=$SEED | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python -c "
import yaml
with open('cifar_params.yaml') as f:
    p = yaml.safe_load(f)
p['poison_rate'] = 0.5
p['dirichlet_alpha'] = 0.5
p['bon_k'] = 1
p['seed'] = $SEED
p['epochs'] = 2000
p['is_poison'] = True
p['defence_method'] = 'ours'
with open('cifar_params_ctrl.yaml', 'w') as f:
    yaml.dump(p, f)
print('Params written.')
"

python main.py --params cifar_params_ctrl.yaml --defense ours \
    2>&1 | tee "$OUT/check7_ctrl_seed${SEED}.log"

echo "Done: $(date)"
