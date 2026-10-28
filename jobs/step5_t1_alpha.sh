#!/bin/bash
#SBATCH --job-name=step5_t1_alpha
#SBATCH --array=1-6
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Tier-1 checker re-run at correct dirichlet_alpha (explicitly set).
# Previous headline runs silently inherited alpha=1.0 from the base YAML.
# This script explicitly sets alpha=0.5 (tasks 1-3) and alpha=0.1 (tasks 4-6).
#
# check12b setup: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, 600 rounds, FilterFL (ours).
# ours_standard explicitly [0.5, 0.4, 0.3, 0.2].
# probe_save_epochs every 50 rounds (50..600) for full s3/margin trajectory.
# wfc_analysis runs after training: s3, fa, margin, mW, mphi (all train mode).
#
# Array:  1=seed1/a0.5  2=seed2/a0.5  3=seed3/a0.5
#         4=seed1/a0.1  5=seed2/a0.1  6=seed3/a0.1

SEEDS=(1 2 3 1 2 3)
ALPHAS=(0.5 0.5 0.5 0.1 0.1 0.1)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=${SEEDS[$IDX]}
ALPHA=${ALPHAS[$IDX]}

ALPHA_TAG=$(echo $ALPHA | sed 's/\.//')   # 05 or 01
echo "step5_t1_alpha task $SLURM_ARRAY_TASK_ID: seed=$SEED alpha=$ALPHA  job=$SLURM_ARRAY_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_t1_alpha_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_checker_a${ALPHA_TAG}_seed${SEED}
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

python3 -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = 600
p['seed']                         = $SEED
p['dirichlet_alpha']              = $ALPHA
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = list(range(50, 601, 50))
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print(f'yaml: seed=$SEED alpha=$ALPHA ours_standard=[0.5,0.4,0.3,0.2] trigger=checker probe_save_epochs=50..600', flush=True)
PYEOF

LOG=$OUT_DIR/step5_t1_a${ALPHA_TAG}_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. adv_local files saved:"
ls $PROBE_DIR/adv_local_*.pt 2>/dev/null | wc -l

echo "Running wfc_analysis (train mode)..."
python3 $REPO/jobs/step5_wfc_analysis.py \
    --trigger   checker \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $SCRATCH_DIR/FilterFL/data/cifar-10-batches-py \
    --rounds    $(seq -s ' ' 50 50 600) \
    --out       $OUT_DIR/step5_wfc_checker_a${ALPHA_TAG}_seed${SEED}.txt \
    2>&1 | tee $OUT_DIR/step5_wfc_checker_a${ALPHA_TAG}_seed${SEED}_analysis.log

echo "Done: seed=$SEED alpha=$ALPHA"
echo "  Log:     $LOG"
echo "  WFC out: $OUT_DIR/step5_wfc_checker_a${ALPHA_TAG}_seed${SEED}.txt"
echo "  Probes:  $PROBE_DIR"
