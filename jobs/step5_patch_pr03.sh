#!/bin/bash
#SBATCH --job-name=step5_patch_pr03
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Patch trigger, poison_rate=0.3, check12b, 600 rounds, seed=1.
# probe_save_epochs every 50 rounds (50..600) for full s3/margin trajectory.
# wfc_analysis (train mode) run after training: s3, fa, margin, mW, mphi.

echo "step5_patch_pr03 job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_patch_pr03_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_patch_pr03
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
p['poison_rate']                  = 0.3
p['epochs']                       = 600
p['seed']                         = 1
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = 'patch'
p['probe_save_epochs']            = list(range(50, 601, 50))
p['probe_save_dir']               = '$PROBE_DIR'
# patch positions already in cifar_params.yaml; do not overwrite
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched: trigger=patch poison_rate=0.3 probe_save_epochs=50..600(step50)', flush=True)
PYEOF

LOG=$OUT_DIR/step5_patch_pr03.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done. Probes in $PROBE_DIR:"
ls $PROBE_DIR/adv_local_*.pt 2>/dev/null | wc -l

echo "Running wfc_analysis (train mode)..."
python3 $REPO/jobs/step5_wfc_analysis.py \
    --trigger   patch \
    --probe-dir $PROBE_DIR \
    --ffl-dir   $SCRATCH_DIR/FilterFL \
    --data-dir  $SCRATCH_DIR/FilterFL/data/cifar-10-batches-py \
    --rounds    $(seq -s ' ' 50 50 600) \
    --out       $OUT_DIR/step5_wfc_patch_pr03.txt \
    2>&1 | tee $OUT_DIR/step5_wfc_patch_pr03_analysis.log

echo "Done."
echo "  Log:     $LOG"
echo "  WFC out: $OUT_DIR/step5_wfc_patch_pr03.txt"
echo "  Probes:  $PROBE_DIR"
