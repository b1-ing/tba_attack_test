#!/bin/bash
#SBATCH --job-name=step5_crossfam
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Cross-family defense comparison: quiet attack (norm cap = 0.90 = 0.8x benign) vs
# FilterFL (probe-based), FLAME (norm-clip-to-median), FLTrust (cosine-trust-weighted).
#
# Structural claim: a norm-capped attack at 0.8x benign median automatically satisfies
# FLAME's clip threshold (already below median) and may also evade FLTrust if its
# direction has non-trivial cosine similarity with clean gradients.
#
# Array: 1=FilterFL(ours)  2=FLAME  3=FLTrust
# Config: checker, alpha=0.5, norm_cap=0.90, seed=1, 600 rounds.

DEFENSES=(dummy ours flame fltrust)
DEF_TAGS=(dummy filterfl flame fltrust)
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}
TARGET_NORM=0.90

echo "step5_crossfam job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE  tag=$DEF_TAG  TARGET_NORM=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_crossfam_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Build FLTrust root dataset (200 images, 20 per class, from test batch) ────
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
if [[ ! -f $ROOT_PKL ]]; then
    python3 -I - <<ROOTEOF
import pickle, numpy as np, sys

data_dir = '$DATA_DIR'
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

with open(f'{data_dir}/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
imgs   = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels = np.array(d[b'labels'])

# 20 images per class
xs, ys = [], []
for c in range(10):
    idx = np.where(labels == c)[0][:20]
    xs.append(imgs[idx])
    ys.extend([c] * len(idx))
x = np.concatenate(xs, axis=0)                          # (200, 3, 32, 32)
x = (x - mean[None, :, None, None]) / std[None, :, None, None]   # normalise
y = np.array(ys, dtype=np.int64)

with open('$ROOT_PKL', 'wb') as f:
    pickle.dump((x, y), f)
print(f'Root dataset saved: {x.shape}  classes={set(ys)}', flush=True)
ROOTEOF
    echo "FLTrust root dataset: $ROOT_PKL"
else
    echo "FLTrust root dataset already exists: $ROOT_PKL"
fi

# ── Patch: norm-cap on adversarial client deltas ──────────────────────────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

cap_block = f'''\
        # [NORMCAP] Cap adversarial client update norms to target_norm.
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_delta_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_delta_norm = _lnc_delta_sq ** 0.5
                if _lnc_delta_norm > _lnc_target:
                    _lnc_scale = _lnc_target / _lnc_delta_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_scale
                        ).to(w_locals[_lnc_i][k].dtype)
                    logger.info(f'[NORMCAP] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_delta_norm:.3f}}->'
                                f'{{_lnc_target:.3f}} scale={{_lnc_scale:.4f}}')
        # [END NORMCAP]
'''

insert_marker = '        # ── FLAME / FLTrust aggregation'
if insert_marker not in src:
    print('ERROR: insert marker not found', flush=True); sys.exit(1)

src = src.replace(insert_marker, cap_block + insert_marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched {alg_path}: norm cap at {tn}', flush=True)
PATCHEOF

if [[ $? -ne 0 ]]; then echo "Patch failed — aborting"; exit 1; fi

# ── YAML ──────────────────────────────────────────────────────────────────────
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
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = [300, 600]
p['probe_save_dir']               = '$OUT_DIR/probes_crossfam_$DEF_TAG'
# FLTrust-specific
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: defense=$DEFENSE checker alpha=0.5 600r cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_crossfam_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo ""
echo "Training done."

# ── Summary ───────────────────────────────────────────────────────────────────
python3 -I - <<SUMEOF
import re

log_path = '$LOG'
with open(log_path) as f:
    lines = f.readlines()

asr_lines   = [l for l in lines if 'Global_Test_Poison:True'  in l]
clean_lines = [l for l in lines if 'Global_Test_Poison:False' in l]
def_lines   = [l for l in lines if 'count_all_attack' in l]
flame_lines = [l for l in lines if 'FLAME ep=' in l]
trust_lines = [l for l in lines if 'FLTrust ep=' in l]

asr_final   = asr_lines[-1].strip()   if asr_lines   else 'N/A'
clean_final = clean_lines[-1].strip() if clean_lines  else 'N/A'

tot_atk = tot_def = cor_def = 0
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)', def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))
det_rate = cor_def / tot_atk if tot_atk > 0 else float('nan')

# FLAME: report median norm at first and last round where it was triggered
flame_summary = f'{len(flame_lines)} rounds logged' if flame_lines else 'N/A (not FLAME run)'
if flame_lines:
    flame_summary += f'  first: {flame_lines[0].strip()}'
    if len(flame_lines) > 1:
        flame_summary += f'  last: {flame_lines[-1].strip()}'

# FLTrust: report mean trust scores at last logged round
trust_summary = 'N/A (not FLTrust run)'
if trust_lines:
    last = trust_lines[-1]
    trust_summary = last.strip()

out = f"""=== cross-family arm: defense=$DEFENSE  norm_cap=$TARGET_NORM ===
Final ASR:        {asr_final}
Final clean:      {clean_final}
Defense counts:   count_all_attack={tot_atk} count_all_defense={tot_def} count_correct_defense={cor_def}
Detection rate:   {cor_def}/{tot_atk} = {det_rate:.3f}
FLAME:            {flame_summary}
FLTrust:          {trust_summary}
"""
print(out)
with open('$OUT_DIR/step5_crossfam_${DEF_TAG}_summary.txt', 'w') as f:
    f.write(out)
SUMEOF

echo "Done."
echo "  Log:     $LOG"
echo "  Summary: $OUT_DIR/step5_crossfam_${DEF_TAG}_summary.txt"
