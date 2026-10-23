#!/bin/bash
#SBATCH --job-name=step5_ffl_probe
#SBATCH --array=1-4
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Step 5 FilterFL eval-mode vs train-mode comparison.
# Adds a probe_eval_mode flag to algorithm.py so the GAN forward pass can
# run in .eval() mode (running stats) rather than .train() mode (batch stats).
#
# Array:
#   1  filterfl_train  checker   (existing behaviour, train mode probe)
#   2  filterfl_train  cov64pos
#   3  filterfl_eval   checker   (new: .eval() before probe forward pass)
#   4  filterfl_eval   cov64pos

MODES=(train train eval eval)
TRIGGERS=(checker cov64pos checker cov64pos)

IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
MODE=${MODES[$IDX]}
TRIGGER=${TRIGGERS[$IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID: mode=$MODE trigger=$TRIGGER"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_ffl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR
OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP_DIR=$HOME/ffl_check1

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
cd $SCRATCH_DIR

# Patch yaml
python3 - <<'PYEOF'
import yaml
with open('FilterFL/cifar_params.yaml') as f:
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
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml patched')
PYEOF

# If eval mode: patch algorithm.py to add .eval() before probe forward pass
if [ "$MODE" == "eval" ]; then
python3 - <<'PYEOF'
import re

path = 'FilterFL/algorithm.py'
with open(path) as f:
    src = f.read()

# Insert ResNet8_model.eval() immediately before the with torch.no_grad() block
# in the probe forward pass (line ~181 in original)
old = '                with torch.no_grad():\n                    out = torch.softmax(ResNet8_model(g_imgs[0]), dim=1)'
new = '                ResNet8_model.eval()  # probe_eval_mode\n                with torch.no_grad():\n                    out = torch.softmax(ResNet8_model(g_imgs[0]), dim=1)'

if old not in src:
    print('ERROR: patch anchor not found in algorithm.py — check indentation')
    import sys; sys.exit(1)
patched = src.replace(old, new, 1)
with open(path, 'w') as f:
    f.write(patched)
print('algorithm.py patched for eval-mode probe')
PYEOF
fi

# Set trigger type in yaml
if [ "$TRIGGER" == "cov64pos" ]; then
python3 - <<'PYEOF'
import yaml
with open('FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
# cov64pos is an additive trigger — need to override trigger_type if supported
# FilterFL's train.py handles trigger_type=checker or trigger_type=badnets_patch etc.
# For cov64pos we rely on the BON/probe mechanism; trigger_type just controls training.
# Set trigger_type to something that allows custom apply_pattern_batch_gpu override.
p['trigger_type'] = 'cov64pos'  # used to label output; actual patch via train.py hook
p['trigger_amplitude'] = 0.5
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('trigger set to cov64pos')
PYEOF
else
python3 - <<'PYEOF'
import yaml
with open('FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['trigger_type'] = 'checker'
p['trigger_amplitude'] = 0.5
with open('FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('trigger set to checker')
PYEOF
fi

# Run the FilterFL framework directly
TAG=filterfl_${MODE}_${TRIGGER}
LOG=$OUT_DIR/step5_${TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

# Extract per-epoch s3 and ASR from the log
python3 - $LOG $OUT_DIR/step5_${TAG}.csv $MODE $TRIGGER <<'PYEOF'
import sys, re, csv

log_path, csv_path, mode, trigger = sys.argv[1:]
rows = []
with open(log_path) as f:
    for line in f:
        # Log lines: "FEAT_ALIGN ep=N client=C fa=F s3=S"
        m = re.search(r'FEAT_ALIGN ep=(\d+) client=(\S+) fa=([0-9.]+) s3=([0-9.]+)', line)
        if m:
            rows.append({'mode': mode, 'trigger': trigger,
                         'epoch': int(m.group(1)), 'client': m.group(2),
                         'feat_align': float(m.group(3)), 's3': float(m.group(4))})

with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['mode','trigger','epoch','client','feat_align','s3'])
    wr.writeheader(); wr.writerows(rows)
print(f'Parsed {len(rows)} FEAT_ALIGN rows → {csv_path}')
PYEOF

echo "Done: $TAG"
