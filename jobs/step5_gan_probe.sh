#!/bin/bash
#SBATCH --job-name=step5_ganprobe
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# FilterFL with ORIGINAL paper GAN probe (gan_defense).
#
# Our published normcap results used pixel_space_defense (our variant).
# The original FilterFL paper uses GAN-generated probes (gan_defense in gan.py).
# Different detectors have different blind spots — the "stricter variant"
# argument is asserted, not measured. This run verifies whether our
# orthogonal normcap adversary also evades the paper's GAN probe.
#
# If evades GAN: result stands with original paper's method (stronger).
# If caught by GAN: result scoped to pixel-space probe variant (must be stated).
#
# Config: cap=0.057, alpha=0.5, seeds 1-3, 600 rounds, defense=ours (gan_defense).
# probe_mode NOT set -> defaults to gan_defense (original paper).
# Matches the YAML of step5_lowcap_2k FilterFL arm except probe_mode removed.
#
# Per-epoch ~30-50s (GAN generator training). Budget: 12h per seed.

SEED=$SLURM_ARRAY_TASK_ID   # seeds 1, 2, 3
TARGET_NORM=0.057
ROUNDS=600

echo "step5_ganprobe job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "seed=$SEED cap=$TARGET_NORM rounds=$ROUNDS probe=gan_defense"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_ganprobe_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3 python not accessible on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap + CLIENT_FLAGS ────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP] Cap adversarial delta to TARGET_NORM; log CLIENT_FLAGS.
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
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]')
        # [END NORMCAP]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'normcap patch applied (cap={tn}): compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML — NO probe_mode set → uses gan_defense (original paper) ─────────────
$PYTHON -I - <<PYEOF
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
p['epochs']                       = $ROUNDS
p['seed']                         = $SEED
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
# probe_mode NOT set -> gan_defense (original FilterFL paper)
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok probe=gan_defense seed=$SEED cap=$TARGET_NORM rounds=$ROUNDS', flush=True)
PYEOF

LOG=$OUT_DIR/step5_ganprobe_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense ours 2>&1 | tee $LOG

echo "Training done."

# ── Summary: ASR + detection rate + s3 comparison ────────────────────────────
$PYTHON -I -W ignore - <<SUMEOF
import re, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

seed = $SEED
cap  = float('$TARGET_NORM')
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ganprobe_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ganprobe_seed{seed}_summary.txt'
CTRL = 0.019

LOG = open(LOG_PATH).read()

# ASR trajectory
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

# Stable implantation
stable = None
for j,(c,a) in enumerate(pairs):
    if a-CTRL > 0.50:
        future = pairs[j+1:j+11]
        if all(fa-CTRL > 0.30 for _,fa in future):
            stable = j+1; break

lines = [f'=== FilterFL GAN probe: seed={seed} cap={cap} probe=gan_defense ===', '']
if pairs:
    fc, fa = pairs[-1]
    lines += [f'  Final: clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  Stable impl eval: {stable}',
              '  Trajectory (every 100 evals):']
    for ep in list(range(0, len(pairs), 100)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
else:
    lines.append('  [no eval rows]')

# Detection counts
def_lines = [l for l in LOG.split('\n') if 'count_all_attack' in l]
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)',
                  def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))
        det_rate = cor_def / tot_atk if tot_atk else float('nan')
        lines += ['', f'  Detections: {cor_def}/{tot_atk} adv rounds = {det_rate:.3f}',
                  f'  (pixel_space_defense had 0/{tot_atk} detections for same cap)']

# Adversary s3 scores (for GAN probe comparison with pixel probe)
# Log format from FilterFL: "client_id, pred_class, score"
# Need to extract adversarial client s3 scores specifically
adv_s3 = re.findall(
    r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\].*?\n((?:.*\n){0,15}?).*count_all', LOG)

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
