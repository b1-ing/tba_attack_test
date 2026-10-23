#!/bin/bash
#SBATCH --job-name=step5_irrev_adp
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Patch irreversibility rerun with adaptive implanting:
# implant until ASR >= 95% for 10 consecutive rounds (cap 300), then 500 benign rounds.

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_irrev_adp_${SLURM_JOB_ID}
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

python3 jobs/step5_irreversibility.py \
    --trigger        patch           \
    --code-dir       $SCRATCH_DIR/FilterFL \
    --warmup-dir     $WARMUP_DIR     \
    --out            $SCRATCH_DIR    \
    --implant-mode   adaptive        \
    --implant-asr-thresh 0.95        \
    --implant-asr-consec 10          \
    --implant-cap    300             \
    --total-rounds   500             \
    --poison-rate    0.5             \
    --seed           1

cp $SCRATCH_DIR/step5_irrev_patch.csv $OUT_DIR/step5_irrev_patch_adaptive.csv 2>/dev/null && \
    echo "Copied irrev_patch_adaptive CSV to $OUT_DIR/"

echo "Done."
