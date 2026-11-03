#!/bin/bash
#SBATCH --job-name=step5_consec
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Dynamic hypothesis: round-to-round direction consistency.
# Measures cos(adv_dir_t, adv_dir_{t-1}) per round for checker, noise, stripes.
# Hypothesis: checker has high consecutive cosine (coherent direction -> accumulates);
# noise/stripes have lower (random direction changes -> cancels under cap).
#
# Array: task 1=checker, 2=noise, 3=stripes. Seed=1, 200 rounds each.
# Also logs pre-cap adv_norm and D(t) proxy: projection of global model delta
# onto the running mean adv direction.

case $SLURM_ARRAY_TASK_ID in
  1) TRIGGER=checker ;;
  2) TRIGGER=noise   ;;
  3) TRIGGER=stripes ;;
esac
SEED=1
TARGET_NORM=0.057
ROUNDS=200
AMPLITUDE=0.1145

echo "step5_consec job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM rounds=$ROUNDS"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_consec_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: normcap + CONSEC (consecutive direction cosine) + OVERLAP + NORM ──
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP+DYNAMIC] Cap adversarial delta; log OVERLAP, pre-cap NORM, and
        # CONSEC (cosine between consecutive adversary update directions).
        import numpy as _np_nc
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        _adv_idx_nc = [i for i, n in enumerate(agent_name_keys) if n in adversarial_name_keys]
        _ben_idx_nc = [i for i, n in enumerate(agent_name_keys) if n not in adversarial_name_keys]

        def _delta_flat_nc(idx):
            return _np_nc.concatenate([
                (w_locals[idx][k] - ori_weight[k]).float().cpu().numpy().flatten()
                for k in _lnc_keys])

        if _adv_idx_nc and _ben_idx_nc:
            _adv_d_nc  = _delta_flat_nc(_adv_idx_nc[0])
            _adv_n_nc  = float(_np_nc.linalg.norm(_adv_d_nc))
            if _adv_n_nc > 1e-9:
                _adv_dir_nc = _adv_d_nc / _adv_n_nc

                # Overlap: cos(adv_dir, mean_ben_dir)
                _ben_dirs_nc  = [_delta_flat_nc(i) for i in _ben_idx_nc]
                _ben_norms_nc = [float(_np_nc.linalg.norm(d)) for d in _ben_dirs_nc]
                _ben_unit_nc  = [d / (n + 1e-9) for d, n in zip(_ben_dirs_nc, _ben_norms_nc)]
                _ben_mean_nc  = _np_nc.mean(_ben_unit_nc, axis=0)
                _bmn_nc = float(_np_nc.linalg.norm(_ben_mean_nc))
                if _bmn_nc > 1e-9:
                    _ov_nc = float(_np_nc.dot(_adv_dir_nc, _ben_mean_nc / _bmn_nc))
                    logger.info(f'[OVERLAP] ep={{epoch}} cos_adv_ben={{_ov_nc:.6f}} adv_norm={{_adv_n_nc:.6f}}')

                # Consecutive direction cosine: cos(adv_dir_t, adv_dir_{{t-1}})
                # Stored in algorithm module globals across rounds.
                import sys as _sys_nc
                _alg_mod_nc = _sys_nc.modules[__name__]
                _prev_dir_nc = getattr(_alg_mod_nc, '_nc_prev_adv_dir', None)
                if _prev_dir_nc is not None and len(_prev_dir_nc) == len(_adv_dir_nc):
                    _consec_nc = float(_np_nc.dot(_adv_dir_nc, _prev_dir_nc))
                    logger.info(f'[CONSEC] ep={{epoch}} cos_t_tm1={{_consec_nc:.6f}}')
                setattr(_alg_mod_nc, '_nc_prev_adv_dir', _adv_dir_nc.copy())

        # Apply norm cap.
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
        # [END NORMCAP+DYNAMIC]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'patch applied (cap={tn}): compiles OK')
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
print('yaml ok trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM rounds=$ROUNDS', flush=True)
PYEOF

LOG=$OUT_DIR/step5_consec_${TRIGGER}_seed${SEED}.log

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

trigger = '$TRIGGER'
seed    = $SEED
cap     = float('$TARGET_NORM')
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_consec_{trigger}_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_consec_{trigger}_seed{seed}_summary.txt'

LOG = open(LOG_PATH).read()

lines = [f'=== consec probe: trigger={trigger} seed={seed} cap={cap} rounds=200 ===', '']

# Pre-cap norms
norms = [float(x) for x in re.findall(r'\[OVERLAP\].*adv_norm=([+-]?[\d.eE+-]+)', LOG)]
if norms:
    lines.append(f'  pre-cap adv_norm: n={len(norms)} mean={np.mean(norms):.4f} '
                 f'std={np.std(norms):.4f} p5={np.percentile(norms,5):.4f} '
                 f'p50={np.percentile(norms,50):.4f} p95={np.percentile(norms,95):.4f}')

# Consecutive direction cosines
consecs = [float(x) for x in re.findall(r'\[CONSEC\].*cos_t_tm1=([+-]?[\d.eE+-]+)', LOG)]
if consecs:
    lines += ['',
              f'  consec cos(dir_t, dir_t-1): n={len(consecs)}',
              f'    mean={np.mean(consecs):.4f}  std={np.std(consecs):.4f}',
              f'    p5={np.percentile(consecs,5):.4f}  p25={np.percentile(consecs,25):.4f}  '
              f'p50={np.percentile(consecs,50):.4f}  p75={np.percentile(consecs,75):.4f}  '
              f'p95={np.percentile(consecs,95):.4f}',
              f'    frac>0.5: {np.mean(np.array(consecs)>0.5):.3f}',
              f'    frac>0.0: {np.mean(np.array(consecs)>0.0):.3f}',
              f'    frac<0.0: {np.mean(np.array(consecs)<0.0):.3f}']
else:
    lines.append('  [No CONSEC rows found]')

# Overlap stats
overlaps = [float(x) for x in re.findall(r'\[OVERLAP\].*cos_adv_ben=([+-]?[\d.eE+-]+)', LOG)]
if overlaps:
    lines += ['',
              f'  cos(adv,ben_mean): n={len(overlaps)} mean={np.mean(overlaps):.4f} '
              f'std={np.std(overlaps):.4f}']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
