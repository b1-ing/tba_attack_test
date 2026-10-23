#!/bin/bash
#SBATCH --job-name=step5_seeds23
#SBATCH --array=2-3
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Seed replication: FilterFL, checker, check12b, 600 rounds.
# Seeds 2 and 3 (array task ID = seed number).
# probe_save_epochs at every 50 rounds (50..600) for full s3/margin trajectory.
# Same config as probespec_chk2 (seed=1) except seed and probe_save_epochs.

SEED=$SLURM_ARRAY_TASK_ID
echo "step5_seeds23 task $SLURM_ARRAY_TASK_ID: seed=$SEED  job=$SLURM_ARRAY_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_seeds23_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_checker_seed${SEED}
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

python3 - <<PYEOF
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
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = list(range(50, 601, 50))
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print(f'yaml patched: seed=$SEED trigger=checker probe_save_epochs=50..600 (step 50)', flush=True)
PYEOF

LOG=$OUT_DIR/step5_seed${SEED}_checker.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. Probes in $PROBE_DIR:"
ls $PROBE_DIR/adv_local_*.pt 2>/dev/null | wc -l

echo "Running wfc_analysis..."
python3 $REPO/jobs/step5_wfc_analysis.py \
    --trigger   checker \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $SCRATCH_DIR/FilterFL/data/cifar-10-batches-py \
    --rounds    $(seq -s ' ' 50 50 600) \
    --out       $OUT_DIR/step5_wfc_checker_seed${SEED}.txt \
    2>&1 | tee $OUT_DIR/step5_wfc_checker_seed${SEED}_analysis.log

echo "Done: seed=$SEED"
echo "  Log:      $LOG"
echo "  WFC out:  $OUT_DIR/step5_wfc_checker_seed${SEED}.txt"
echo "  Probes:   $PROBE_DIR"
