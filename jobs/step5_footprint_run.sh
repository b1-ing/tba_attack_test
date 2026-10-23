#!/bin/bash
#SBATCH --job-name=step5_footprint
#SBATCH --array=1-2
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Steps 2a + 2b: weight-footprint paired training.
# Array task 1 = checker, task 2 = patch

TRIGGERS=(checker patch)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: trigger=$TRIGGER"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_footprint_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP_DIR=$HOME/ffl_check1

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
cp -r $REPO/jobs     $SCRATCH_DIR/

# Use pre-staged CIFAR-10 to avoid downloading
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 data staged"

cd $SCRATCH_DIR

# Patch yaml for check12b setup
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
p['poison_rate']                  = 0.5
p['seed']                         = 1
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched', flush=True)
PYEOF

python3 jobs/step5_footprint_run.py \
    --trigger    $TRIGGER \
    --code-dir   $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out        $SCRATCH_DIR \
    --n-rounds   60 \
    --seed       1 \
    --alpha      0.5 \
    --amplitude  0.5

# Copy outputs
cp $SCRATCH_DIR/step5_footprint_${TRIGGER}.csv         $OUT_DIR/ 2>/dev/null && echo "Copied ${TRIGGER} CSV"
cp $SCRATCH_DIR/step5_footprint_${TRIGGER}_paired.csv  $OUT_DIR/ 2>/dev/null && echo "Copied ${TRIGGER} paired CSV"
cp $SCRATCH_DIR/step5_footprint_${TRIGGER}_d.npy       $OUT_DIR/ 2>/dev/null && echo "Copied ${TRIGGER} d_hat npy"

echo "Done: trigger=$TRIGGER"
