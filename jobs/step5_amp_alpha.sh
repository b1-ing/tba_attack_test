#!/bin/bash
#SBATCH --job-name=step5_amp_alpha
#SBATCH --array=1-12
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Amplitude sweep re-run at correct dirichlet_alpha (explicitly set).
# Triggers: checker/eps4/eps8/eps16/eps32/eps64 — same checkerboard pattern,
# amplitude-only variation. seed=1. 600 rounds.
# ours_standard explicitly [0.5, 0.4, 0.3, 0.2].
# probe_save_epochs=[50,150,400,600] for wfc decomposition + ASR eval.
#
# Array tasks 1-6:  alpha=0.5 (checker/eps4/eps8/eps16/eps32/eps64)
# Array tasks 7-12: alpha=0.1 (checker/eps4/eps8/eps16/eps32/eps64)

TRIGGERS=(checker eps4 eps8 eps16 eps32 eps64  checker eps4 eps8 eps16 eps32 eps64)
ALPHAS=(0.5 0.5 0.5 0.5 0.5 0.5  0.1 0.1 0.1 0.1 0.1 0.1)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}
ALPHA=${ALPHAS[$IDX]}

ALPHA_TAG=$(echo $ALPHA | sed 's/\.//')   # 05 or 01
echo "step5_amp_alpha task $SLURM_ARRAY_TASK_ID: trigger=$TRIGGER alpha=$ALPHA  job=$SLURM_ARRAY_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_amp_alpha_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_${TRIGGER}_a${ALPHA_TAG}
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
p['seed']                         = 1
p['dirichlet_alpha']              = $ALPHA
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = '$TRIGGER'
if '$TRIGGER' == 'checker':
    p['trigger_amplitude_raw']    = 0.1145
p['probe_save_epochs']            = [50, 150, 400, 600]
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print(f'yaml: trigger=$TRIGGER alpha=$ALPHA ours_standard=[0.5,0.4,0.3,0.2]', flush=True)
PYEOF

LOG=$OUT_DIR/step5_amp_${TRIGGER}_a${ALPHA_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. adv_local files:"
ls $PROBE_DIR/adv_local_*.pt 2>/dev/null | wc -l

echo "Running wfc_analysis (train mode)..."
python3 $REPO/jobs/step5_wfc_analysis.py \
    --trigger   $TRIGGER \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $SCRATCH_DIR/FilterFL/data/cifar-10-batches-py \
    --rounds    50 150 400 600 \
    --out       $OUT_DIR/step5_wfc_${TRIGGER}_a${ALPHA_TAG}.txt \
    2>&1 | tee $OUT_DIR/step5_wfc_${TRIGGER}_a${ALPHA_TAG}_analysis.log

echo "Running ASR eval..."
python3 $REPO/jobs/step5_amp_asr.py \
    --trigger   $TRIGGER \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $SCRATCH_DIR/FilterFL/data/cifar-10-batches-py \
    --out       $OUT_DIR/step5_amp_asr_${TRIGGER}_a${ALPHA_TAG}.txt \
    2>&1 | tee -a $LOG

echo "Done: trigger=$TRIGGER alpha=$ALPHA"
echo "  Log:     $LOG"
echo "  WFC out: $OUT_DIR/step5_wfc_${TRIGGER}_a${ALPHA_TAG}.txt"
echo "  ASR out: $OUT_DIR/step5_amp_asr_${TRIGGER}_a${ALPHA_TAG}.txt"
echo "  Probes:  $PROBE_DIR"
