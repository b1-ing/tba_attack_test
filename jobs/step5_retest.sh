#!/bin/bash
#SBATCH --job-name=step5_retest
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-6
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Re-run all triggers with the FIXED test.py (apply_additive_norm for all non-patch types).
# Tasks: 1=checker, 2=blend_87, 3=blend_75, 4=blend_25, 5=stripes, 6=low_col_opt.
# All seed=1, normcap=0.057, 600 rounds.
# Purpose: verify that "NO" results were a test artifact (test never applied trigger).

case $SLURM_ARRAY_TASK_ID in
  1) TRIGGER=checker    ;;
  2) TRIGGER=blend_87   ;;
  3) TRIGGER=blend_75   ;;
  4) TRIGGER=blend_25   ;;
  5) TRIGGER=stripes    ;;
  6) TRIGGER=low_col_opt ;;
esac
SEED=1
TARGET_NORM=0.057
ROUNDS=600
AMPLITUDE=0.1145

echo "step5_retest job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM"

if [[ $TRIGGER == "low_col_opt" ]]; then
    [[ -f $HOME/ffl_step5/low_col_opt_delta.npy ]] || {
        echo "FATAL: low_col_opt_delta.npy not found"; exit 1; }
fi

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_retest_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: python not found"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap only ───────────────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP] Cap adversarial delta to {tn}.
        import numpy as _np_nc
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_norm = _lnc_sq ** 0.5
                if _lnc_norm > _lnc_target:
                    _sc = _lnc_target / _lnc_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] + (w_locals[_lnc_i][k] - ori_weight[k]) * _sc
                        ).to(w_locals[_lnc_i][k].dtype)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} '
                    f'order=[{{",".join(str(n) for n in agent_name_keys)}}] '
                    f'adv_mask=[{{",".join("1" if n in adversarial_name_keys else "0" for n in agent_name_keys)}}]')
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

# ── YAML ─────────────────────────────────────────────────────────────────────
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
p['trigger_type']                 = '$TRIGGER'
p['trigger_amplitude_raw']        = $AMPLITUDE
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_retest_${TRIGGER}_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense ours 2>&1 | tee $LOG

echo "Training done."

# ── Summary ───────────────────────────────────────────────────────────────────
$PYTHON -I -W ignore - <<SUMEOF
import re, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

seed    = $SEED
trigger = '$TRIGGER'
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_retest_{trigger}_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_retest_{trigger}_seed{seed}_summary.txt'

with open(LOG_PATH) as f:
    LOG = f.read()

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs, i = [], 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

stable = None
for j,(c,a) in enumerate(pairs):
    if a-CTRL > 0.50:
        future = pairs[j+1:j+11]
        if all(fa-CTRL > 0.30 for _,fa in future):
            stable = j+1; break

result = 'YES' if stable else 'NO'
lines = [f'=== RETEST {trigger} seed={seed} (FIXED test.py) ===', '']
if pairs:
    fc, fa = pairs[-1]
    f100_net = np.mean([max(0, a-CTRL) for _,a in pairs[-100:]])
    lines += [f'  RESULT: {result}',
              f'  Final:  clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  F100 mean net_asr: {f100_net:.3f}',
              f'  Stable impl eval: {stable}',
              '',
              '  ASR trajectory (every 100 evals):']
    for ep in list(range(0, len(pairs), 100)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
else:
    lines.append('  [no eval rows found]')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
