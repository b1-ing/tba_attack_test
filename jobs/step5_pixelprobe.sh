#!/bin/bash
#SBATCH --job-name=step5_pixelprobe
#SBATCH --array=1-2
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Experiment 1: Generator-swap test (Option A — pixel-space probes).
# Replaces GAN probe generation with direct pixel-space optimisation
# using the same Stage 1/2 objective. Stage 3 exclusion rule unchanged.
#
# Array:
#   1  checker  — primary: does detection persist past round 300?
#   2  patch    — regression check (should remain detected)
#
# check12b setup: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.
# Controls logged per round:
#   ADV S3:   FEAT_ALIGN lines
#   BEN S3:   FEAT_ALIGN_BEN lines (log_ben_s3=True)
#   FP rate:  from count_all_defense / count_correct_defense lines

TRIGGERS=(checker patch)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
TRIGGER=${TRIGGERS[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: probe_mode=pixel trigger=$TRIGGER"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_pixelprobe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/

mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# Patch YAML: check12b params + pixel probe + per-round benign S3 logging
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
p['trigger_amplitude_raw']        = 0.1145
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched: probe_mode=pixel log_ben_s3=True trigger=$TRIGGER', flush=True)
PYEOF

TAG=pixelprobe_${TRIGGER}
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

# Extract per-epoch S3 for ADV and BEN clients
python3 - $LOG $OUT_DIR/step5_${TAG}_s3.csv <<'PYEOF'
import sys, re, csv

log_path, csv_path = sys.argv[1:]
rows = []
with open(log_path) as f:
    for line in f:
        # ADV S3: "FEAT_ALIGN ep=N client=C fa=F s3=S"
        m = re.search(r'FEAT_ALIGN ep=(\d+) client=(\S+) fa=([0-9.eE+-]+) s3=([0-9.eE+-]+)', line)
        if m:
            rows.append({'type': 'ADV', 'epoch': int(m.group(1)),
                         'client': m.group(2), 'fa': float(m.group(3)),
                         's3': float(m.group(4))})
            continue
        # BEN S3: "FEAT_ALIGN_BEN ep=N client=C fa=F s3=S"
        m = re.search(r'FEAT_ALIGN_BEN ep=(\d+) client=(\S+) fa=([0-9.eE+-]+) s3=([0-9.eE+-]+)', line)
        if m:
            rows.append({'type': 'BEN', 'epoch': int(m.group(1)),
                         'client': m.group(2), 'fa': float(m.group(3)),
                         's3': float(m.group(4))})

with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['type', 'epoch', 'client', 'fa', 's3'])
    wr.writeheader(); wr.writerows(rows)
print(f'Parsed {len(rows)} S3 rows (ADV+BEN) → {csv_path}', flush=True)
PYEOF

# Extract defense counts per epoch for FP rate
python3 - $LOG $OUT_DIR/step5_${TAG}_defense.csv <<'PYEOF'
import sys, re, csv

log_path, csv_path = sys.argv[1:]
rows = []
# "count_all_attack: N, count_all_defense: M, count_correct_defense: K"
for line in open(log_path):
    m = re.search(r'count_all_attack:\s*(\d+),\s*count_all_defense:\s*(\d+),\s*count_correct_defense:\s*(\d+)', line)
    if m:
        rows.append({'count_all_attack': int(m.group(1)),
                     'count_all_defense': int(m.group(2)),
                     'count_correct_defense': int(m.group(3))})
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['count_all_attack','count_all_defense','count_correct_defense'])
    wr.writeheader(); wr.writerows(rows)
print(f'Parsed {len(rows)} defense-count rows → {csv_path}', flush=True)
PYEOF

echo "Done: probe_mode=pixel trigger=$TRIGGER"
echo "Logs: $LOG"
echo "S3 CSV: $OUT_DIR/step5_${TAG}_s3.csv"
