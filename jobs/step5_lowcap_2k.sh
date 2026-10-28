#!/bin/bash
#SBATCH --job-name=step5_lc2k
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Floor-cap long run: cap=0.012 (1% benign norm), 2400 rounds.
#
# Question: is the ~0% ASR at cap=0.012 in 600 rounds a patience artifact
# (accumulation needs more rounds) or a real threshold below which no
# implantation occurs?
#
# If linear accumulation: cap*rounds = constant for ASR ceiling.
#   normcap: 0.057 * 600 * 1.0 adv/rd = 34.2 "dose units"
#   floor:   0.012 * 2400 * 1.0 adv/rd = 28.8 "dose units"
# These are comparable -> ASR should eventually reach ~98% (patience artifact).
#
# If real threshold: a minimum per-round contribution is needed; below it,
# noise overwhelms signal -> ASR plateaus below the ceiling.
#
# Arms:
#   1: FedAvg  (no defense, purest signal)
#   2: FilterFL (same cap, with defense overhead)

DEFENSES=(dummy None ours)
DEF_TAGS=(dummy fedavg filterfl)
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

TARGET_NORM=0.012    # 1% of benign norm ~1.2
ROUNDS=2400
SEED=1

echo "step5_lc2k job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "cap=$TARGET_NORM rounds=$ROUNDS defense=$DEFENSE"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_lc2k_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

# Fail fast: anaconda3 not accessible on this node
[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3/2024.10 python3.12 not accessible on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap + CLIENT_FLAGS ─────────────────────────────────────────────
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
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok defense=$DEFENSE rounds=$ROUNDS cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_lc2k_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Post-training analysis ─────────────────────────────────────────────────────
$PYTHON -I - <<'PYEOF2'
import re, sys
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

LOG      = open('/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_lc2k_' +
               '__DEFTAG__' + '.log').read()
OUT_FILE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_lc2k_' + '__DEFTAG__' + '_summary.txt'
CTRL = 0.019

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

lines = ['=== Floor-cap long run: cap=__CAP__ defense=__DEFTAG__ rounds=__ROUNDS__ ===', '']
if pairs:
    fc, fa = pairs[-1]
    lines += [f'  Final: clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  Mean clean: {np.mean([c for c,a in pairs]):.3f}',
              '  Trajectory (every 200 evals):']
    for ep in list(range(0, len(pairs), 200)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:5d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
    # Find implantation round (first eval where net_asr > 50%)
    impl = next((ep+1 for ep,(c,a) in enumerate(pairs) if a-CTRL > 0.5), None)
    lines.append(f'  Implantation round (net_asr>50%): {impl}')
else:
    lines.append('  [no eval rows]')

adv_counts = [m.group(1).count('1')
              for m in re.finditer(r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\]', LOG)]
if adv_counts:
    ac = np.array(adv_counts)
    total_adv = np.sum(ac)
    dose = float('__CAP__') * total_adv / 10.0
    lines += ['', f'  Total adv appearances: {int(total_adv)}/{len(ac)} rounds',
              f'  Cumulative dose (cap * appearances / 10): {dose:.3f}',
              f'  Compare normcap dose (0.057*~600*1.0/10): 3.42']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
PYEOF2

# Replace placeholders in the inline python with actual values
sed -i "s/__DEFTAG__/$DEF_TAG/g; s/__CAP__/$TARGET_NORM/g; s/__ROUNDS__/$ROUNDS/g" \
    $SCRATCH_DIR/FilterFL/*.py 2>/dev/null || true

$PYTHON -I - <<PYEOF3
import re, sys
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

def_tag  = '$DEF_TAG'
cap      = float('$TARGET_NORM')
rounds   = int('$ROUNDS')
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_lc2k_{def_tag}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_lc2k_{def_tag}_summary.txt'
CTRL     = 0.019

LOG = open(LOG_PATH).read()

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

lines = [f'=== Floor-cap long run: cap={cap} defense={def_tag} rounds={rounds} ===', '']
if pairs:
    fc, fa = pairs[-1]
    lines += [f'  Final: clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  Mean clean: {np.mean([c for c,a in pairs]):.3f}',
              '  Trajectory (every 200 evals):']
    for ep in list(range(0, len(pairs), 200)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:5d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
    impl = next((ep+1 for ep,(c,a) in enumerate(pairs) if a-CTRL > 0.5), None)
    lines.append(f'  Implantation round (net_asr>50%): {impl}')
else:
    lines.append('  [no eval rows]')

adv_counts = [m.group(1).count('1')
              for m in re.finditer(r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\]', LOG)]
if adv_counts:
    ac = np.array(adv_counts)
    total_adv = int(np.sum(ac))
    dose = cap * total_adv / 10.0
    lines += ['', f'  Total adv appearances: {total_adv}/{len(ac)} rounds',
              f'  Cumulative dose (cap * appearances / 10): {dose:.3f}',
              f'  Compare normcap dose (0.057*~600*1.0/10): 3.42']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
PYEOF3

echo "Done. Log: $LOG"
