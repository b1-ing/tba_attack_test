#!/bin/bash
#SBATCH --job-name=step5_dose
#SBATCH --array=1-6
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Dose-response: FedAvg × {checker, patch} × poison_rate ∈ {0.05, 0.10, 0.20}
# Array index → (trigger, poison_rate):
#   1  checker  0.05
#   2  checker  0.10
#   3  checker  0.20
#   4  patch    0.05
#   5  patch    0.10
#   6  patch    0.20

TRIGGERS=(checker checker checker patch patch patch)
RATES=(0.05 0.10 0.20 0.05 0.10 0.20)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}
PRATE=${RATES[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: trigger=$TRIGGER poison_rate=$PRATE"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_dose_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
    --poison-rate $PRATE     \
    --code-dir   $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out        $SCRATCH_DIR \
    --n-rounds   300 \
    --seed       1

cp $SCRATCH_DIR/step5_*.csv $OUT_DIR/ 2>/dev/null || true
cp $SCRATCH_DIR/step5_*_ckpt.pt $OUT_DIR/ 2>/dev/null || true

echo "Done: trigger=$TRIGGER poison_rate=$PRATE"
