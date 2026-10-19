#!/bin/bash --login
#SBATCH --job-name=ffl_check12
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check12_%j.out
#SBATCH --exclude=node871

# Check 12: Full 2000-round FilterFL run with the checkerboard trigger.
# Same protocol as check7_ctrl (rate=0.5, alpha=0.5, seed=1) but with
# trigger_type=checker, amplitude=0.5.
#
# Expected: ASR stays high (checker backdoor works) while detection rate
# drops to near zero (Stage-3 blind to checker's high-frequency feature direction).
# Check7 ctrl baseline: ASR=2.10%, detection=98.9%, FP=259 at epoch 2000.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

SEED=${SEED:-1}

OUT=$HOME/ffl_check12
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check12_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | trigger=checker amp=0.5 | seed=$SEED | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python -c "
import yaml
with open('cifar_params.yaml') as f:
    p = yaml.safe_load(f)
p['poison_rate']       = 0.5
p['dirichlet_alpha']   = 0.5
p['bon_k']             = 1
p['seed']              = $SEED
p['epochs']            = 2000
p['is_poison']         = True
p['defence_method']    = 'ours'
p['trigger_type']      = 'checker'
p['trigger_amplitude'] = 0.5
with open('cifar_params_check12.yaml', 'w') as f:
    yaml.dump(p, f)
print('Params written.')
"

python main.py --params cifar_params_check12.yaml --defense ours \
    2>&1 | tee "$OUT/check12_checker_seed${SEED}.log"

echo "Done: $(date)"
