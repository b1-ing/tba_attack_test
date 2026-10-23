#!/bin/bash
#SBATCH --job-name=step5_wfc_sweep
#SBATCH --array=1-7
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# W_fc drift sweep: FilterFL probe-spectrum run for 7 implanting triggers
# (checker and patch already have probes_checker2/ and probes_patch2/).
# Each task saves adv_local_ep{N}.pt at rounds 50/150/330/400 for margin decomp.
#
# check12b setup: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds, FilterFL (ours).
#
# Array mapping:
#  1=stripes  2=checker_energy_matched  3=noise  4=eps4
#  5=eps8     6=corner4x4              7=corner4x4_energy_matched

TRIGGERS=(stripes checker_energy_matched noise eps4 eps8 corner4x4 corner4x4_energy_matched)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}

echo "step5_wfc_sweep task $SLURM_ARRAY_TASK_ID: trigger=$TRIGGER  job=$SLURM_ARRAY_JOB_ID"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_wfc_sweep_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_${TRIGGER}
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
p['seed']                         = 1
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = '$TRIGGER'
p['probe_save_epochs']            = [50, 150, 330, 400]
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: trigger=$TRIGGER probe_save_epochs=[50,150,330,400]', flush=True)
PYEOF

LOG=$OUT_DIR/step5_wfc_${TRIGGER}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. Files in $PROBE_DIR:"
ls $PROBE_DIR/*.pt 2>/dev/null | head -20

# Run margin decomposition immediately on completion
python3 $REPO/jobs/step5_wfc_analysis.py \
    --trigger  $TRIGGER \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
    --out       $OUT_DIR/step5_wfc_${TRIGGER}.txt \
    2>&1 | tee -a $LOG

echo "Done: trigger=$TRIGGER"
echo "  Probes: $PROBE_DIR"
echo "  Log:    $LOG"
