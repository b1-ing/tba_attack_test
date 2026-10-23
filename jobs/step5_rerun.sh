#!/bin/bash
#SBATCH --job-name=step5_rerun
#SBATCH --array=1-16
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Step 5 GPU reruns: FedAvg / FLTrust / FoolsGold / FLAME (corrected)
# 4 defenses × 3 triggers = 12 arms + 3 near-IID control arms (α=100, FedAvg)
#
# Array index → (defense, trigger, alpha):
#   1  fedavg     checker   0.5
#   2  fedavg     cov64pos  0.5
#   3  fedavg     patch     0.5
#   4  fltrust    checker   0.5
#   5  fltrust    cov64pos  0.5
#   6  fltrust    patch     0.5
#   7  foolsgold  checker   0.5
#   8  foolsgold  cov64pos  0.5
#   9  foolsgold  patch     0.5
#  10  flame      checker   0.5
#  11  flame      cov64pos  0.5
#  12  flame      patch     0.5
#  13  fedavg     checker   100  (near-IID α=100 control)
#  14  fltrust    checker   100  (near-IID α=100 control)
#  15  foolsgold  checker   100  (near-IID α=100 control)
#  16  flame      checker   100  (near-IID α=100 control)

DEFENSES=(fedavg fedavg fedavg fltrust fltrust fltrust foolsgold foolsgold foolsgold flame flame flame fedavg fltrust foolsgold flame)
TRIGGERS=(checker cov64pos patch checker cov64pos patch checker cov64pos patch checker cov64pos patch checker checker checker checker)
ALPHAS=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 100 100 100 100)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
DEFENSE=${DEFENSES[$IDX]}
TRIGGER=${TRIGGERS[$IDX]}
ALPHA=${ALPHAS[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: defense=$DEFENSE trigger=$TRIGGER alpha=$ALPHA"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP_DIR=$HOME/ffl_check1

source $VENV/bin/activate
cd $REPO

# Copy FilterFL code to scratch (avoid NFS contention)
cp -r FilterFL $SCRATCH_DIR/
cp -r jobs     $SCRATCH_DIR/

# Stage CIFAR-10 to avoid 30-45min download
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

cd $SCRATCH_DIR

# Override yaml at check12b settings before running
python3 - <<'PYEOF'
import yaml, sys
with open('FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = 600
p['seed']                         = 1
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched', flush=True)
PYEOF

python3 jobs/step5_rerun.py \
    --defense  $DEFENSE   \
    --trigger  $TRIGGER   \
    --alpha    $ALPHA     \
    --code-dir $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out      $SCRATCH_DIR  \
    --n-rounds 600 \
    --seed     1   \
    --checkpoint-every 50

# Copy results back
cp $SCRATCH_DIR/step5_*.csv $OUT_DIR/ 2>/dev/null || true
cp $SCRATCH_DIR/step5_*_ckpt.pt $OUT_DIR/ 2>/dev/null || true

echo "Done. Results in $OUT_DIR"
ls -lh $OUT_DIR/step5_*
