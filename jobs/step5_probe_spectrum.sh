#!/bin/bash
#SBATCH --job-name=step5_probespectrum
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Probe-spectrum instrumented rerun — checker arm only.
# Saves pixel-space probe tensors + model states at rounds 50, 150, 250, 330, 400
# so the analysis script can measure checker-band power and compute GAN reference.
#
# check12b setup: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.
# probe_mode=pixel, log_ben_s3=True, probe_save_epochs=[50,150,250,330,400]

echo "step5_probespectrum job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_probespectrum_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_checker
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# Patch YAML: check12b + pixel probe + probe saving at key rounds
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
print('yaml patched: probe_mode=pixel probe_save_epochs=[50,150,250,330,400]', flush=True)
PYEOF

TAG=probespectrum_checker
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Training done. Running spectrum analysis..."

python3 $REPO/jobs/step5_probe_spectrum_analysis.py \
    $PROBE_DIR \
    $SCRATCH_DIR/FilterFL \
    2>&1 | tee $OUT_DIR/step5_${TAG}_spectrum.txt

# Also extract S3 CSV (same as pixelprobe script)
python3 - $LOG $OUT_DIR/step5_${TAG}_s3.csv <<'PYEOF'
import sys, re, csv
log_path, csv_path = sys.argv[1:]
rows = []
with open(log_path) as f:
    for line in f:
        m = re.search(r'FEAT_ALIGN ep=(\d+) client=(\S+) fa=([0-9.eE+-]+) s3=([0-9.eE+-]+)', line)
        if m:
            rows.append({'type': 'ADV', 'epoch': int(m.group(1)),
                         'client': m.group(2), 'fa': float(m.group(3)),
                         's3': float(m.group(4))})
            continue
        m = re.search(r'FEAT_ALIGN_BEN ep=(\d+) client=(\S+) fa=([0-9.eE+-]+) s3=([0-9.eE+-]+)', line)
        if m:
            rows.append({'type': 'BEN', 'epoch': int(m.group(1)),
                         'client': m.group(2), 'fa': float(m.group(3)),
                         's3': float(m.group(4))})
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['type','epoch','client','fa','s3'])
    wr.writeheader(); wr.writerows(rows)
print(f'Parsed {len(rows)} S3 rows → {csv_path}', flush=True)
PYEOF

echo "Done. Probes: $PROBE_DIR"
echo "Spectrum: $OUT_DIR/step5_${TAG}_spectrum.txt"
echo "Log: $LOG"
