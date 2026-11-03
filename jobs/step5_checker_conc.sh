#!/bin/bash
#SBATCH --job-name=step5_chkconc
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Short 80-round checker+normcap run with [CONC] logging.
# Purpose: measure weight-space norm concentration for checker to compare with
# noise (0.96) and stripes (0.98) from step5_noise_stripes_probe runs.
# Uses seed=1 only. 80 rounds enough to get stable per-round concentration values.

SEED=1
TARGET_NORM=0.057
ROUNDS=80
TRIGGER=checker
AMPLITUDE=0.1145

echo "step5_chkconc job $SLURM_JOB_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM rounds=$ROUNDS probe=gan_defense"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_chkconc_${SLURM_JOB_ID}
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

# ── Patch: normcap + CONC + OVERLAP ──────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP+METRICS] Cap adversarial delta; log OVERLAP and weight-space CONC.
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

                # Overlap
                _ben_dirs_nc = [_delta_flat_nc(i) for i in _ben_idx_nc]
                _ben_norms_nc = [float(_np_nc.linalg.norm(d)) for d in _ben_dirs_nc]
                _ben_unit_nc  = [d / (n + 1e-9) for d, n in zip(_ben_dirs_nc, _ben_norms_nc)]
                _ben_mean_nc  = _np_nc.mean(_ben_unit_nc, axis=0)
                _bmn_nc = float(_np_nc.linalg.norm(_ben_mean_nc))
                if _bmn_nc > 1e-9:
                    _ov_nc = float(_np_nc.dot(_adv_dir_nc, _ben_mean_nc / _bmn_nc))
                    logger.info(f'[OVERLAP] ep={{epoch}} cos_adv_ben={{_ov_nc:.6f}} adv_norm={{_adv_n_nc:.6f}}')

                # Weight-space norm concentration
                _adv_sq_nc = _adv_d_nc ** 2
                _total_sq_nc = float(_adv_sq_nc.sum())
                if _total_sq_nc > 1e-12:
                    _sorted_sq_nc = _np_nc.sort(_adv_sq_nc)[::-1]
                    _cumsum_nc = _np_nc.cumsum(_sorted_sq_nc)
                    _n_params_nc = len(_adv_d_nc)
                    _concs_nc = {{}}
                    for _pct_nc in [0.001, 0.01, 0.05, 0.10]:
                        _k_nc = max(1, int(_n_params_nc * _pct_nc))
                        _concs_nc[_pct_nc] = float(_cumsum_nc[_k_nc-1] / _total_sq_nc)
                    logger.info(
                        f'[CONC] ep={{epoch}} n={{_n_params_nc}} '
                        f'top0.1pct={{_concs_nc[0.001]:.4f}} '
                        f'top1pct={{_concs_nc[0.01]:.4f}} '
                        f'top5pct={{_concs_nc[0.05]:.4f}} '
                        f'top10pct={{_concs_nc[0.10]:.4f}}'
                    )

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
        # [END NORMCAP+METRICS]
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

LOG=$OUT_DIR/step5_checker_conc_seed${SEED}.log

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

CTRL    = 0.019
LOG_PATH = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_checker_conc_seed1.log'
OUT_FILE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_checker_conc_seed1_summary.txt'

LOG = open(LOG_PATH).read()

lines = ['=== checker conc probe: seed=1 cap=0.057 rounds=80 probe=gan_defense ===', '']

# Overlap stats
overlaps = re.findall(r'\[OVERLAP\] ep=\d+ cos_adv_ben=([+-]?[\d.eE+-]+)', LOG)
if overlaps:
    ov = [float(x) for x in overlaps]
    lines += [f'  cos(adv,ben_mean): n={len(ov)} mean={np.mean(ov):.4f} std={np.std(ov):.4f} '
              f'p5={np.percentile(ov,5):.4f} p50={np.percentile(ov,50):.4f} p95={np.percentile(ov,95):.4f}']

# Norm concentration stats (weight-space)
for pct_label in ['top0.1pct', 'top1pct', 'top5pct', 'top10pct']:
    vals = re.findall(rf'\[CONC\].*{pct_label}=([+-]?[\d.eE+-]+)', LOG)
    if vals:
        v = [float(x) for x in vals]
        lines.append(f'  wt-conc {pct_label}: n={len(v)} mean={np.mean(v):.4f} std={np.std(v):.4f} '
                     f'min={np.min(v):.4f} max={np.max(v):.4f}')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
