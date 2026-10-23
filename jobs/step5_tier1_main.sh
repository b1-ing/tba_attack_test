#!/bin/bash
#SBATCH --job-name=step5_tier1
#SBATCH --array=1-11
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Step 3 Tier 1: FedAvg × 11 triggers, 300 rounds, poison_rate=0.5, seed=1
# Array index → trigger variant:
#  1=stripes  2=checker  3=checker_energy_matched  4=noise  5=eps4
#  6=eps8  7=sin16  8=sin32  9=corner4x4  10=corner4x4_energy_matched  11=patch

TRIGGERS=(stripes checker checker_energy_matched noise eps4 eps8 sin16 sin32 corner4x4 corner4x4_energy_matched patch)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: trigger=$TRIGGER"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_tier1_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP_DIR=$HOME/ffl_check1

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
cp -r $REPO/jobs     $SCRATCH_DIR/

mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

cd $SCRATCH_DIR

python3 - <<'PYEOF'
import yaml
with open('FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['seed']                         = 1
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched', flush=True)
PYEOF

python3 jobs/step5_rerun.py \
    --defense    fedavg      \
    --trigger    $TRIGGER    \
    --poison-rate 0.5        \
    --code-dir   $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out        $SCRATCH_DIR \
    --n-rounds   300 \
    --seed       1

cp $SCRATCH_DIR/step5_*.csv $OUT_DIR/ 2>/dev/null || true

echo "Done: trigger=$TRIGGER"
