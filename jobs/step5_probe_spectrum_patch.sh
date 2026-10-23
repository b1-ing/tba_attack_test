#!/bin/bash
#SBATCH --job-name=step5_probespec_patch
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Probe-spectrum instrumented rerun — PATCH arm.
# Mirrors step5_probe_spectrum.sh (checker) but with trigger_type='patch'.
# Saves pixel-space probe tensors + d1/d2 model states at rounds 50,150,250,330,400
# so step5_Htest_analysis_v3.py can run the frozen-probe H test on the patch arm.
#
# check12b: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.
# probe_mode=pixel, probe_save_epochs=[50,150,250,330,400]

echo "step5_probespec_patch job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_probespec_patch_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_patch
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# Patch YAML: check12b + patch trigger + pixel probe + probe saving
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
p['trigger_type']                 = 'patch'
p['probe_save_epochs']            = [50, 150, 250, 330, 400]
p['probe_save_dir']               = '$PROBE_DIR'
# patch positions already in YAML (0_poison_pattern); do not overwrite
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched: trigger_type=patch probe_mode=pixel probe_save_epochs=[50,150,250,330,400]',
      flush=True)
PYEOF

TAG=probespectrum_patch
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. Probe files in $PROBE_DIR:"
ls $PROBE_DIR/*.pt 2>/dev/null || echo "(none)"

# Run frozen-probe H analysis on the patch arm immediately
echo ""
echo "Running H-test analysis (patch arm)..."
python3 $REPO/jobs/step5_Htest_analysis_v3.py \
    $PROBE_DIR \
    $SCRATCH_DIR/FilterFL \
    patch \
    2>&1 | tee $OUT_DIR/step5_Htest_analysis_v3_patch.txt

echo "Done."
echo "  Probes:   $PROBE_DIR"
echo "  H-table:  $OUT_DIR/step5_Htest_analysis_v3_patch.txt"
echo "  Log:      $LOG"
