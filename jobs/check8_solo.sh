#!/bin/bash --login
#SBATCH --job-name=ffl_check8_solo
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-0:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check8_solo_%j.out

# Check 8 (Client-Selection, Solo): 2000-round FilterFL run where ONLY client 34
# (lowest class-0 share: 0.043) poisons, even though all 10 adversarial clients
# are nominally compromised. Tests whether restricting to the lowest-share client
# keeps global model contamination low enough for Stage-3 to stay below rho.
#
# adversarial_whitelist = [34]  → only client 34 injects poison each round.
# All other clients in adversarial_namelist train as benign.
#
# Expected appearances per round: 34/(10% of 100) ≈ 0.1/round → ~200 total.
# Compare ASR and detection rate against check7_ctrl (all 10 adversarial).

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

WORK=$SCRATCH/ffl_check8_solo_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
mkdir -p saved_models
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | whitelist=[34] (solo) | seed=$SEED | start: $(date)"
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
p['adversarial_whitelist'] = [34]
with open('cifar_params_check8_solo.yaml', 'w') as f:
    yaml.dump(p, f)
print('Params written.')
"

python main.py --params cifar_params_check8_solo.yaml --defense ours \
    2>&1 | tee "$OUT/check8_solo_seed${SEED}.log"

echo "Done: $(date)"
