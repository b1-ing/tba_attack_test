#!/bin/bash --login
#SBATCH --job-name=ffl_check8_trio
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check8_trio_%j.out

# Check 8 (Client-Selection, Trio): 2000-round FilterFL run where only the three
# adversarial clients with the lowest live-loop Stage-3 scores (in the full-10
# check7_ctrl run) are activated: clients 34, 47, 99.
#   Client 34: class-0 share=0.043, mean live s3=0.576 (best — sometimes evades)
#   Client 47: class-0 share=0.015, mean live s3=0.690
#   Client 99: class-0 share=unknown, mean live s3=0.652
#
# adversarial_whitelist = [34, 47, 99]
# Intermediate between solo (1 client) and full (10 clients).
# Expected adversarial appearances per round ≈ 3/10 ≈ 0.3/round → ~600 total.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data

SEED=${SEED:-1}
OUT=$HOME/ffl_check8
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check8_trio_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | whitelist=[34,47,99] (trio) | seed=$SEED | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python -c "
import yaml
with open('cifar_params.yaml') as f:
    p = yaml.safe_load(f)
p['poison_rate'] = 0.5
p['dirichlet_alpha'] = 0.5
p['seed'] = $SEED
p['epochs'] = 2000
p['is_poison'] = True
p['defence_method'] = 'ours'
p['adversarial_whitelist'] = [34, 47, 99]
with open('cifar_params_check8_trio.yaml', 'w') as f:
    yaml.dump(p, f)
print('Params written.')
"

python main.py --params cifar_params_check8_trio.yaml --defense ours \
    2>&1 | tee "$OUT/check8_trio_seed${SEED}.log"

echo "Done: $(date)"
