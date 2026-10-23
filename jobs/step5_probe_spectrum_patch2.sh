#!/bin/bash
#SBATCH --job-name=step5_probespec_pat2
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Probe-spectrum instrumented rerun — PATCH arm, v2.
# Fixes: algorithm.py now saves w_locals[i] as adv_local_ep{N:04d}.pt at probe_save_epochs.
# Saves to probes_patch2/ (probes_patch/ has old incomplete data with no adv_local files).
#
# check12b: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.
# probe_mode=pixel, probe_save_epochs=[50,150,250,330,400]
# log_ben_s3=True  -> FEAT_ALIGN_BEN lines for F-ben analysis

echo "step5_probespec_pat2 job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_probespec_pat2_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_patch2
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
# patch positions already in cifar_params.yaml; do not overwrite
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched: trigger_type=patch probe_mode=pixel probe_save_epochs=[50,150,250,330,400] log_ben_s3=True',
      flush=True)
PYEOF

TAG=probespectrum_patch2
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done. Files in $PROBE_DIR:"
ls $PROBE_DIR/*.pt 2>/dev/null | head -20

# ── H-test v3: frozen probe, train mode ──────────────────────────────────────
echo ""
echo "Running H-test v3 (patch arm, train mode)..."
python3 $REPO/jobs/step5_Htest_analysis_v3.py \
    $PROBE_DIR \
    $SCRATCH_DIR/FilterFL \
    patch \
    2>&1 | tee $OUT_DIR/step5_Htest_analysis_v3_patch2.txt

# ── BN mode check: train vs eval s3 ─────────────────────────────────────────
echo ""
echo "Running BN mode check (patch arm)..."
python3 $REPO/jobs/step5_bn_mode_check.py \
    $PROBE_DIR \
    $SCRATCH_DIR/FilterFL \
    2>&1 | tee $OUT_DIR/step5_bn_mode_check_patch2.txt

# ── FEAT_ALIGN extraction for scatter ───────────────────────────────────────
echo ""
echo "Extracting FEAT_ALIGN lines for F-adv scatter..."
grep 'FEAT_ALIGN ep=' $LOG | grep -v 'FEAT_ALIGN_BEN' \
    > $OUT_DIR/step5_patch2_fa_adv.txt
grep 'FEAT_ALIGN_BEN ep=' $LOG \
    > $OUT_DIR/step5_patch2_fa_ben.txt
echo "  ADV lines: $(wc -l < $OUT_DIR/step5_patch2_fa_adv.txt)"
echo "  BEN lines: $(wc -l < $OUT_DIR/step5_patch2_fa_ben.txt)"

echo ""
echo "Done."
echo "  Probes:      $PROBE_DIR"
echo "  Log:         $LOG"
echo "  H-table:     $OUT_DIR/step5_Htest_analysis_v3_patch2.txt"
echo "  BN check:    $OUT_DIR/step5_bn_mode_check_patch2.txt"
echo "  ADV FA:      $OUT_DIR/step5_patch2_fa_adv.txt"
echo "  BEN FA:      $OUT_DIR/step5_patch2_fa_ben.txt"
