#!/bin/bash
#SBATCH --job-name=step5_probespec_chk2
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Probe-spectrum instrumented rerun — CHECKER arm, v2.
# Fixes: algorithm.py now saves the ADVERSARIAL CLIENT'S LOCAL MODEL (w_locals[i])
# at probe_save_epochs rounds as adv_local_ep{N:04d}.pt. Previously only the
# global models (ori_weight, w_glob) were saved as d1/d2 in model_ep{N}.pt.
#
# check12b: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.
# probe_mode=pixel, probe_save_epochs=[50,150,250,330,400]

echo "step5_probespec_chk2 job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_probespec_chk2_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_checker2     # separate dir; probes_checker/ has old (wrong) d1 saves
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# Patch YAML
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
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = [50, 150, 250, 330, 400]
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched: trigger=checker probe_mode=pixel probe_save_epochs=[50,150,250,330,400]',
      flush=True)
PYEOF

TAG=probespectrum_chk2
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. Files in $PROBE_DIR:"
ls $PROBE_DIR/*.pt 2>/dev/null | head -20

# Run H-test v3 analysis immediately
echo ""
echo "Running H-test v3 analysis (checker arm)..."
python3 $REPO/jobs/step5_Htest_analysis_v3.py \
    $PROBE_DIR \
    $SCRATCH_DIR/FilterFL \
    checker \
    2>&1 | tee $OUT_DIR/step5_Htest_analysis_v3_checker.txt

echo "Done."
echo "  Probes/models: $PROBE_DIR"
echo "  H-table:       $OUT_DIR/step5_Htest_analysis_v3_checker.txt"
echo "  Log:           $LOG"
