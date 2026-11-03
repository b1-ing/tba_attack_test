#!/bin/bash
#SBATCH --job-name=step5_vproj
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Measure backdoor direction v = (Δ_adv − mean_ben)/||…|| per round (ep≥50).
# Projects all client deltas onto v; logs attacker push and benign sum.
# Prediction: stripes → proj_ben_sum systematically negative (benign erodes);
#             checker / low_col_opt → proj_ben_sum ≈ 0 (neutral) → accumulates.
# Tasks: 1=checker, 2=stripes, 3=low_col_opt. Seed=1, cap=0.057, 600 rounds.

case $SLURM_ARRAY_TASK_ID in
  1) TRIGGER=checker    ;;
  2) TRIGGER=stripes    ;;
  3) TRIGGER=low_col_opt ;;
esac
SEED=1
TARGET_NORM=0.057
ROUNDS=600
AMPLITUDE=0.1145

echo "step5_vproj job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM"

if [[ $TRIGGER == "low_col_opt" ]]; then
    [[ -f $HOME/ffl_step5/low_col_opt_delta.npy ]] || {
        echo "FATAL: low_col_opt_delta.npy not found"; exit 1; }
fi

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_vproj_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: normcap + V_PROJ ───────────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [V_PROJ + NORMCAP]
        import numpy as _np_vp
        _lnc_target = {tn}
        _pkeys_vp = [k for k in ori_weight
                     if ori_weight[k].dtype != torch.int64
                     and 'running_' not in k and 'num_batches' not in k]
        _all_keys_vp = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        _adv_idx_vp = [i for i,n in enumerate(agent_name_keys) if n in adversarial_name_keys]
        _ben_idx_vp = [i for i,n in enumerate(agent_name_keys) if n not in adversarial_name_keys]

        if epoch >= 50 and _adv_idx_vp and _ben_idx_vp:
            # Compute pre-normcap adversary delta and benign deltas (learnable params only)
            _d_adv_vp = _np_vp.concatenate([
                (w_locals[_adv_idx_vp[0]][k].float().cpu().numpy()
                 - ori_weight[k].float().cpu().numpy()).flatten()
                for k in _pkeys_vp])
            _ben_ds_vp = []
            for _bii in _ben_idx_vp:
                _bd = _np_vp.concatenate([
                    (w_locals[_bii][k].float().cpu().numpy()
                     - ori_weight[k].float().cpu().numpy()).flatten()
                    for k in _pkeys_vp])
                _ben_ds_vp.append(_bd)
            _mean_ben_vp = _np_vp.mean(_ben_ds_vp, axis=0)

            # Backdoor direction: v = (Δ_adv − mean_ben) / ||…||
            _v_raw_vp   = _d_adv_vp - _mean_ben_vp
            _v_norm_vp  = float(_np_vp.linalg.norm(_v_raw_vp))
            _adv_norm_vp = float(_np_vp.linalg.norm(_d_adv_vp))
            if _v_norm_vp > 1e-9:
                _v_vp = _v_raw_vp / _v_norm_vp
                _proj_adv_vp = float(_np_vp.dot(_d_adv_vp, _v_vp))
                _proj_ben_each_vp = [float(_np_vp.dot(_bd, _v_vp)) for _bd in _ben_ds_vp]
                _proj_ben_sum_vp  = sum(_proj_ben_each_vp)
                _proj_ben_mean_vp = _proj_ben_sum_vp / max(1, len(_proj_ben_each_vp))
                logger.info(
                    f'[V_PROJ] ep={{epoch}} '
                    f'proj_adv={{_proj_adv_vp:.6f}} '
                    f'proj_ben_sum={{_proj_ben_sum_vp:.6f}} '
                    f'proj_ben_mean={{_proj_ben_mean_vp:.6f}} '
                    f'adv_norm={{_adv_norm_vp:.4f}} '
                    f'v_norm={{_v_norm_vp:.4f}}'
                )

        # ── Normcap ──────────────────────────────────────────────────────────
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _all_keys_vp)
                _lnc_nrm = _lnc_sq ** 0.5
                if _lnc_nrm > _lnc_target:
                    _sc = _lnc_target / _lnc_nrm
                    for k in _all_keys_vp:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] + (w_locals[_lnc_i][k] - ori_weight[k]) * _sc
                        ).to(w_locals[_lnc_i][k].dtype)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} '
                    f'adv_mask=[{{",".join("1" if n in adversarial_name_keys else "0" for n in agent_name_keys)}}]')
        # [END V_PROJ + NORMCAP]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'V_PROJ+normcap patch applied (cap={tn}): compiles OK')
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

LOG=$OUT_DIR/step5_vproj_${TRIGGER}_seed${SEED}.log

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
N_SEL   = 10   # clients selected per round
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_vproj_{trigger}_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_vproj_{trigger}_seed{seed}_summary.txt'

with open(LOG_PATH) as f:
    LOG = f.read()

# Parse V_PROJ lines
rows = re.findall(
    r'\[V_PROJ\] ep=(\d+) proj_adv=([+-]?[\d.eE+-]+) proj_ben_sum=([+-]?[\d.eE+-]+) '
    r'proj_ben_mean=([+-]?[\d.eE+-]+) adv_norm=([+-]?[\d.eE+-]+) v_norm=([+-]?[\d.eE+-]+)',
    LOG)

eps       = np.array([int(r[0])   for r in rows])
proj_adv  = np.array([float(r[1]) for r in rows])
proj_ben_sum  = np.array([float(r[2]) for r in rows])
proj_ben_mean = np.array([float(r[3]) for r in rows])
adv_norms = np.array([float(r[4]) for r in rows])
v_norms   = np.array([float(r[5]) for r in rows])

# Normcap scale applied to adversary each round
nc_scale  = np.minimum(1.0, cap / np.maximum(adv_norms, 1e-9))
# Effective attacker push on global model (in v direction, scaled by FedAvg weight 1/N_SEL)
proj_adv_eff = proj_adv * nc_scale / N_SEL
# Benign contribution to global model (in v direction, FedAvg weight 1/N_SEL per client, summed)
proj_ben_eff = proj_ben_sum / N_SEL
# Net accumulation per round in v direction
net_round = proj_adv_eff + proj_ben_eff

# ASR trajectory
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs, i = [], 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2
final_asr = pairs[-1][1] if pairs else float('nan')

lines = [
    f'=== V_PROJ probe: {trigger} seed={seed} cap={cap} ===',
    f'  Rounds with v_proj logged (ep>=50): {len(eps)}',
    f'  Final ASR: {final_asr:.3f}',
    '',
    '  Backdoor direction projections (mean ± std, ep>=50):',
    f'  proj_adv  (pre-normcap, raw):  mean={np.mean(proj_adv):.4f}  std={np.std(proj_adv):.4f}',
    f'  adv_norm  (pre-normcap):       mean={np.mean(adv_norms):.4f}',
    f'  nc_scale  (cap/adv_norm):      mean={np.mean(nc_scale):.4f}',
    '',
    '  --- Effective contributions to global model (scaled by 1/N_SEL) ---',
    f'  proj_adv_eff (attacker push):  mean={np.mean(proj_adv_eff):.6f}  std={np.std(proj_adv_eff):.6f}',
    f'  proj_ben_eff (benign sum/N):   mean={np.mean(proj_ben_eff):.6f}  std={np.std(proj_ben_eff):.6f}',
    f'  net_round    (adv+ben):        mean={np.mean(net_round):.6f}  std={np.std(net_round):.6f}',
    f'  sign(proj_ben_eff):            {np.sign(np.mean(proj_ben_eff)):+.0f}  '
    f'(frac_negative={np.mean(proj_ben_eff < 0):.3f})',
    f'  ratio |benign|/|attacker|:    {abs(np.mean(proj_ben_eff)) / (abs(np.mean(proj_adv_eff))+1e-12):.4f}',
    '',
    '  Per-epoch sample (every 100 rounds, ep>=50):',
]
sel = eps[::100]
for e in sel:
    mask = eps == e
    if mask.any():
        pa = float(np.mean(proj_adv_eff[mask]))
        pb = float(np.mean(proj_ben_eff[mask]))
        lines.append(f'    ep={e:4d}  adv_eff={pa:+.6f}  ben_eff={pb:+.6f}  net={pa+pb:+.6f}')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
