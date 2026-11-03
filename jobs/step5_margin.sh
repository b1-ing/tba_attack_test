#!/bin/bash
#SBATCH --job-name=step5_margin
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Inference-margin probe: log Δ_bd and Δ_nat per round for each trigger.
# Δ_bd  = logit_current[t](x+δ) − logit_clean[t](x+δ)   (backdoor contribution)
# Δ_nat = max_{c≠t} logit_clean[c](x+δ)                  (natural competitor in clean model)
# Prediction: Δ_bd ≈ equal; Δ_nat large for stripes/truck_grad, small for checker/low_col_opt.
# Tasks: 1=checker, 2=stripes, 3=low_col_opt, 4=truck_grad (off-diagonal collinearity breaker).

case $SLURM_ARRAY_TASK_ID in
  1) TRIGGER=checker     ;;
  2) TRIGGER=stripes     ;;
  3) TRIGGER=low_col_opt ;;
  4) TRIGGER=truck_grad  ;;
esac
SEED=1
TARGET_NORM=0.057
ROUNDS=600
AMPLITUDE=0.1145

echo "step5_margin job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_margin_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
CLEAN_MODEL=$OUT_DIR/step5_clean_resnet8.pt
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: python not found"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Truck_grad pre-compute ────────────────────────────────────────────────────
# For task 4 only: compute gradient-based trigger from trained clean model.
# delta = normalize(mean_x grad_x max_{c!=0} logit_c(x)) * 0.1145 in raw [0,1] space.
if [[ $TRIGGER == "truck_grad" ]]; then
    TG_PATH=$OUT_DIR/truck_grad_delta.npy
    if [[ -f $TG_PATH ]]; then
        echo "truck_grad_delta.npy already exists, using it"
    else
        echo "Computing truck_grad trigger from $CLEAN_MODEL ..."
        $PYTHON -W ignore - <<PGEOF
import sys, os
sys.path.insert(0, '$SCRATCH_DIR/FilterFL')
import torch, numpy as np, pickle

TARGET = 0  # airplane
AMPLITUDE = 0.1145
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

from models.ResNet8 import ResNet8
model = ResNet8(num_classes=10)
sd = torch.load('$CLEAN_MODEL', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()

with open('$DATA_DIR/test_batch', 'rb') as f:
    td = pickle.load(f, encoding='bytes')
imgs_raw = td[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels   = np.array(td[b'labels'])
imgs_sel = imgs_raw[labels != TARGET][:300]
imgs_norm = torch.tensor((imgs_sel - MEAN) / STD, dtype=torch.float32)

imgs_var = imgs_norm.clone().detach().requires_grad_(True)
logits = model(imgs_var)
logits_m = logits.clone()
logits_m[:, TARGET] = -1e9
logits_m.max(dim=1).values.mean().backward()

grad = imgs_var.grad.data.mean(0).numpy()  # (3,32,32) normalized space
std_arr = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
grad_raw = grad / std_arr  # convert to raw space
l2 = float(np.linalg.norm(grad_raw))
delta = (grad_raw / l2 * AMPLITUDE).astype(np.float32)
np.save('$TG_PATH', delta)
print(f'truck_grad saved: L2_raw={np.linalg.norm(delta):.4f} max={delta.max():.4f} min={delta.min():.4f}')
PGEOF
        [[ $? -ne 0 ]] && { echo "truck_grad pre-compute failed"; exit 1; }
    fi
fi

if [[ $TRIGGER == "low_col_opt" ]]; then
    [[ -f $OUT_DIR/low_col_opt_delta.npy ]] || { echo "FATAL: low_col_opt_delta.npy not found"; exit 1; }
fi

# ── Patch: normcap + margin probe ─────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM TRIGGER AMPLITUDE
CLEAN_MODEL_PATH=$CLEAN_MODEL
export CLEAN_MODEL_PATH
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch      = os.environ['SCRATCH_DIR']
tn           = float(os.environ['TARGET_NORM'])
trigger      = os.environ['TRIGGER']
amplitude    = float(os.environ['AMPLITUDE'])
clean_path   = os.environ['CLEAN_MODEL_PATH']
data_file    = f'{scratch}/FilterFL/data/cifar-10-batches-py/test_batch'
alg_path     = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# ── Patch 1: normcap + margin init (before FLAME marker) ──────────────────────
patch1 = f'''\
        # [MARGIN_PROBE]
        import sys as _sys_mg, numpy as _np_mg, copy as _copy_mg, pickle as _pkl_mg
        _mg = getattr(_sys_mg, '_mg_state', None)
        if _mg is None:
            _mg = {{}}
            setattr(_sys_mg, '_mg_state', _mg)
        if epoch == 1 and 'clean_model' not in _mg:
            import torch as _tt
            from models.ResNet8 import ResNet8 as _R8
            _cm = _R8(num_classes=10)
            _sd = _tt.load('{clean_path}', map_location='cpu', weights_only=False)
            _cm.load_state_dict(_sd, strict=True)
            _cm.eval()
            _mg['clean_model'] = _cm
            with open('{data_file}', 'rb') as _ff:
                _td = _pkl_mg.load(_ff, encoding='bytes')
            _ir = _td[b'data'].reshape(-1, 3, 32, 32).astype(_np_mg.float32) / 255.0
            _lb = _np_mg.array(_td[b'labels'])
            _sel = _ir[_lb != handle.target][:200]
            _mn = _np_mg.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            _st = _np_mg.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            _itn = (_sel - _mn) / _st
            from trigger_utils import make_raw_delta, apply_additive_norm
            _dn = make_raw_delta('{trigger}', amplitude={amplitude})
            _it = _tt.tensor(_itn, dtype=_tt.float32)
            _mg['triggered_imgs'] = apply_additive_norm(_it, _dn)
            logger.info(f'[MARGIN_INIT] ep={{epoch}} trigger={trigger} n={{len(_mg["triggered_imgs"])}}')
        _nc_tgt = {tn}
        _nc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _nc_i, _nc_nm in enumerate(agent_name_keys):
            if _nc_nm in adversarial_name_keys:
                _nc_sq = sum((w_locals[_nc_i][k] - ori_weight[k]).float().pow(2).sum().item() for k in _nc_keys)
                _nc_nr = _nc_sq ** 0.5
                if _nc_nr > _nc_tgt:
                    _sc = _nc_tgt / _nc_nr
                    for k in _nc_keys:
                        w_locals[_nc_i][k] = (ori_weight[k] + (w_locals[_nc_i][k] - ori_weight[k]) * _sc).to(w_locals[_nc_i][k].dtype)
        # [END MARGIN_PROBE]
'''

marker1 = '        # ── FLAME / FLTrust aggregation'
if marker1 not in src:
    print('ERROR: FLAME marker not found'); sys.exit(1)
src = src.replace(marker1, patch1 + marker1, 1)

# ── Patch 2: margin evaluation (after model update) ────────────────────────────
patch2 = '''\
        # [MARGIN_EVAL]
        import sys as _sys_mg2
        _mg2 = getattr(_sys_mg2, '_mg_state', {})
        if 'clean_model' in _mg2 and epoch % 10 == 0:
            _dev2 = next(handle.model.parameters()).device
            _ti2  = _mg2['triggered_imgs'].to(_dev2)
            _cm2  = _mg2['clean_model'].to(_dev2)
            _tgt2 = handle.target
            handle.model.eval()
            _cm2.eval()
            with torch.no_grad():
                _lc2  = _cm2(_ti2)
                _lcur = handle.model(_ti2)
            handle.model.train()
            _lcc2 = _lc2.clone()
            _lcc2[:, _tgt2] = -1e9
            _dnat = _lcc2.max(dim=1).values.mean().item()
            _dbd  = (_lcur[:, _tgt2] - _lc2[:, _tgt2]).mean().item()
            logger.info(f'[MARGIN] ep={epoch} delta_bd={_dbd:.4f} delta_nat={_dnat:.4f} verdict={int(_dbd > _dnat)}')
        # [END MARGIN_EVAL]
'''

marker2 = '        handle.model.load_state_dict(w_glob)'
if marker2 not in src:
    print('ERROR: load_state_dict marker not found'); sys.exit(1)
src = src.replace(marker2, marker2 + '\n' + patch2, 1)

with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'margin probe patch applied (trigger={trigger} cap={tn}): compiles OK')
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

LOG=$OUT_DIR/step5_margin_${TRIGGER}_seed${SEED}.log

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
cap     = float('$TARGET_NORM')
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_margin_{trigger}_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_margin_{trigger}_seed{seed}_summary.txt'

with open(LOG_PATH) as f:
    LOG = f.read()

# Parse MARGIN lines
rows = re.findall(r'\[MARGIN\] ep=(\d+) delta_bd=([+-]?[\d.eE+-]+) delta_nat=([+-]?[\d.eE+-]+) verdict=([01])', LOG)
eps     = np.array([int(r[0])   for r in rows])
d_bd    = np.array([float(r[1]) for r in rows])
d_nat   = np.array([float(r[2]) for r in rows])
verdicts= np.array([int(r[3])   for r in rows])

# ASR
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs, i = [], 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2
final_asr = pairs[-1][1] if pairs else float('nan')
stable = None
for j, (c, a) in enumerate(pairs):
    if a - CTRL > 0.50:
        future = pairs[j+1:j+11]
        if all(fa - CTRL > 0.30 for _, fa in future):
            stable = j + 1; break
implant = 'YES' if stable else 'NO'

lines = [
    f'=== MARGIN probe: {trigger} seed={seed} cap={cap} ===',
    f'  Final ASR: {final_asr:.3f}  implant={implant}',
    f'  N margin rows: {len(rows)}',
    '',
    '  Δ_bd (mean ± std):  ' + (f'{np.mean(d_bd):+.4f} ± {np.std(d_bd):.4f}' if len(d_bd) else 'N/A'),
    '  Δ_nat (mean ± std): ' + (f'{np.mean(d_nat):+.4f} ± {np.std(d_nat):.4f}' if len(d_nat) else 'N/A'),
    f'  frac(Δ_bd > Δ_nat): {verdicts.mean():.3f}' if len(verdicts) else '  frac: N/A',
    '',
    '  Per-epoch sample (every 50 rounds):',
]
sel_eps = [e for e in range(10, int(eps.max())+1, 50)] if len(eps) else []
for e in sel_eps:
    m = eps == e
    if m.any():
        lines.append(f'    ep={e:4d}  Δ_bd={float(np.mean(d_bd[m])):+.4f}  Δ_nat={float(np.mean(d_nat[m])):+.4f}  v={int(np.mean(verdicts[m])>=0.5)}')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
