#!/bin/bash
#SBATCH --job-name=step5_rflame
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Real FLAME vs normcap (cap=0.057, alpha=0.5, seeds 1-3).
#
# Question: does our orthogonal normcap adversary evade or get caught by
# HDBSCAN cosine clustering?
# - Caught -> boundary case: 'evades magnitude/trust/distance; caught by directional clustering'
# - Evades -> strictly stronger: beats full FLAME incl. clustering
#
# This uses the real FLAME implementation (flame_full_aggregate in algorithm.py):
#   1. HDBSCAN cosine clustering (min_cs = n//2+1, precomputed cosine dist)
#   2. Median norm clip on kept deltas
#   3. Gaussian noise N(0, (lambda*S_t)^2), lambda=0.001 (paper default)
#
# Per-round logs: [FLAME_FULL] labels, kept/excl clients, adv_kept/adv_excl, S_t, sigma
# Post-run summary: adv exclusion rate, final ASR, clean acc trajectory

SEED=$SLURM_ARRAY_TASK_ID   # seeds 1, 2, 3
TARGET_NORM=0.057
LAMBDA=0.001
ROUNDS=600

echo "step5_rflame job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "seed=$SEED cap=$TARGET_NORM lambda=$LAMBDA rounds=$ROUNDS"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_rflame_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3/2024.10 python3.12 not accessible on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# Verify sklearn.cluster.HDBSCAN works on this node
$PYTHON -I -W ignore -c "
from sklearn.cluster import HDBSCAN
import numpy as np
labels = HDBSCAN(min_cluster_size=6, metric='precomputed', allow_single_cluster=True).fit_predict(
    np.zeros((10, 10), dtype=np.float64))
print('HDBSCAN check OK')
" 2>/dev/null || { echo "FATAL: sklearn.cluster.HDBSCAN not working on $(hostname)"; exit 1; }

# ── Patch: normcap ────────────────────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP] Cap adversarial delta to TARGET_NORM; log CLIENT_FLAGS for analysis.
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

import py_compile
try:
    py_compile.compile(alg_path, doraise=True)
    print(f'normcap patch applied (cap={tn}): compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ──────────────────────────────────────────────────────────────────────
$PYTHON -I - <<PYEOF
import yaml, warnings
warnings.filterwarnings('ignore')
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
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
p['log_ben_s3']                   = False
p['flame_lambda']                 = $LAMBDA
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok defense=flame_full seed=$SEED cap=$TARGET_NORM lambda=$LAMBDA', flush=True)
PYEOF

LOG=$OUT_DIR/step5_rflame_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense flame_full 2>&1 | tee $LOG

echo "Training done. Log: $LOG"

# ── Post-run summary ──────────────────────────────────────────────────────────
$PYTHON -I -W ignore - <<PYEOF2
import re, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

seed     = $SEED
cap      = float('$TARGET_NORM')
lam      = float('$LAMBDA')
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_rflame_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_rflame_seed{seed}_summary.txt'
CTRL     = 0.019

LOG = open(LOG_PATH).read()

# ASR trajectory
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

lines = [f'=== Real FLAME: seed={seed} cap={cap} lambda={lam} ===', '']
if pairs:
    fc, fa = pairs[-1]
    lines += [f'  Final: clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  Mean clean: {np.mean([c for c,a in pairs]):.3f}',
              '  Trajectory (every 50 evals):']
    for ep in list(range(0, len(pairs), 50)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
    impl = next((ep+1 for ep,(c,a) in enumerate(pairs) if a-CTRL > 0.5), None)
    lines.append(f'  Stable implantation round (net_asr>50%): {impl}')
else:
    lines.append('  [no eval rows found]')

# FLAME_FULL cluster log parsing
# Format: [FLAME_FULL] ep=X labels=[...] kept=[...] excl=[...] adv_kept=[...] adv_excl=[...]
flame_rows = re.findall(
    r'\[FLAME_FULL\] ep=(\d+) labels=(\[.*?\]) kept=(\[.*?\]) excl=(\[.*?\]) adv_kept=(\[.*?\]) adv_excl=(\[.*?\])',
    LOG)
if flame_rows:
    total_rounds = len(set(r[0] for r in flame_rows))
    adv_excl_rounds = sum(1 for r in flame_rows if r[5] != '[]')
    adv_kept_rounds = total_rounds - adv_excl_rounds
    lines += ['',
              f'  HDBSCAN clustering over {total_rounds} rounds:',
              f'    Adversary EXCLUDED: {adv_excl_rounds} rounds ({100*adv_excl_rounds/max(1,total_rounds):.1f}%)',
              f'    Adversary KEPT (evaded): {adv_kept_rounds} rounds ({100*adv_kept_rounds/max(1,total_rounds):.1f}%)']
    # Sample exclusion detail
    excl_sample = [(r[0], r[5]) for r in flame_rows if r[5] != '[]'][:5]
    if excl_sample:
        lines.append('    First exclusion events (ep, adv_excl):')
        for ep, ae in excl_sample:
            lines.append(f'      ep={ep}: adv_excl={ae}')

# S_t trajectory
st_rows = re.findall(r'\[FLAME_FULL\] ep=(\d+) S_t=([\d.]+) sigma=([\d.]+)', LOG)
if st_rows:
    st_vals = [float(r[1]) for r in st_rows]
    lines += ['', f'  S_t (median kept norm): p5={np.percentile(st_vals,5):.4f} '
              f'p50={np.percentile(st_vals,50):.4f} p95={np.percentile(st_vals,95):.4f}']

# Adversary norm distribution from CLIENT_FLAGS
adv_counts = [m.group(1).count('1')
              for m in re.finditer(r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\]', LOG)]
if adv_counts:
    ac = np.array(adv_counts)
    total_adv = int(np.sum(ac))
    dose = cap * total_adv / 10.0
    lines += ['', f'  Adversary appearances: {total_adv}/{len(ac)} rounds',
              f'  Cumulative dose (cap*appearances/10): {dose:.3f}']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
PYEOF2

echo "Done."
