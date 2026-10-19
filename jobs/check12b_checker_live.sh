#!/bin/bash --login
#SBATCH --job-name=ffl_check12b
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-6:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check12b_%j.out
#SBATCH --exclude=node871

# Check 12b: Full 2000-round FilterFL with checker trigger.
# Fixes vs check12:
#   1. test.py eval applies checker trigger (not patch) → Poison:True = real ASR
#   2. algorithm.py logs FEAT_ALIGN per adversarial client per round → traces mechanism
#
# Baseline comparison: check7_ctrl_seed1 — patch trigger, rate=0.5
#   Final: clean=85.27%, ASR=2.10%, detection=98.9%
# Expectation: clean similar, ASR high (>50%), detection ~10% in steady state

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

SEED=${SEED:-1}

OUT=$HOME/ffl_check12b
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check12b_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | trigger=checker amp=0.5 seed=$SEED | start: $(date)"
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
with open('cifar_params_check12b.yaml', 'w') as f:
    yaml.dump(p, f)
print('Params written.')
"

python main.py --params cifar_params_check12b.yaml --defense ours \
    2>&1 | tee "$OUT/check12b_checker_seed${SEED}.log"

echo "Done: $(date)"
