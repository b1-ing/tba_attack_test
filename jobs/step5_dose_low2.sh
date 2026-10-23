#!/bin/bash
#SBATCH --job-name=step5_dose_low2
#SBATCH --array=1-2
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Dose-response extension: checker × {0.005, 0.002}, FedAvg, 300 rounds
# Array 1 = 0.005, Array 2 = 0.002

RATES=(0.005 0.002)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
PRATE=${RATES[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: trigger=checker poison_rate=$PRATE"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_dose_low2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
    --trigger    checker     \
    --poison-rate $PRATE     \
    --code-dir   $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out        $SCRATCH_DIR \
    --n-rounds   300 \
    --seed       1

cp $SCRATCH_DIR/step5_*.csv $OUT_DIR/ 2>/dev/null || true

echo "Done: checker poison_rate=$PRATE"
