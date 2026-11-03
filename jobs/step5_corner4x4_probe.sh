#!/bin/bash
#SBATCH --job-name=step5_corner4x4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# 2x2 manifold experiment: corner4x4 trigger (LOCALISED + OFF-MANIFOLD).
#
# corner4x4 = +-amplitude checkerboard in top-left 4x4 pixels only.
# Localised like patch, but high-frequency (off-manifold) like checker.
# DECISIVE CELL: if it persists under normcap -> property is off-manifold,
# not spatial spread. Kills "checker is special."
#
# Logs [OVERLAP] ep=X cos_adv_ben=Y per round (uncapped adv dir vs mean ben dir).
# Overlap measurement is the update-space proxy for off/on-manifold.
#
# Cap=0.057, FilterFL gan_defense, seeds 1-3, 600 rounds.

SEED=$SLURM_ARRAY_TASK_ID
TARGET_NORM=0.057
ROUNDS=600
TRIGGER=corner4x4
AMPLITUDE=0.1145

echo "step5_corner4x4 job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "seed=$SEED cap=$TARGET_NORM trigger=$TRIGGER amplitude=$AMPLITUDE probe=gan_defense"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_corner4x4_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: normcap + overlap logging ─────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP+OVERLAP] Cap adversarial delta; log CLIENT_FLAGS and adv/ben overlap.
        import numpy as _np_nc
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        # Compute uncapped adversary direction and mean benign direction for overlap.
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
                _ben_dirs_nc = [_delta_flat_nc(i) for i in _ben_idx_nc]
                _ben_norms_nc = [float(_np_nc.linalg.norm(d)) for d in _ben_dirs_nc]
                _ben_unit_nc  = [d / (n + 1e-9) for d, n in zip(_ben_dirs_nc, _ben_norms_nc)]
                _ben_mean_nc  = _np_nc.mean(_ben_unit_nc, axis=0)
                _bmn_nc = float(_np_nc.linalg.norm(_ben_mean_nc))
                if _bmn_nc > 1e-9:
                    _overlap_nc = float(_np_nc.dot(_adv_dir_nc, _ben_mean_nc / _bmn_nc))
                    logger.info(f'[OVERLAP] ep={{epoch}} cos_adv_ben={{_overlap_nc:.6f}} adv_norm={{_adv_n_nc:.6f}}')

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
        # [END NORMCAP+OVERLAP]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'normcap+overlap patch applied (cap={tn}): compiles OK')
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
# probe_mode NOT set -> gan_defense (original FilterFL paper)
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok trigger=$TRIGGER amplitude=$AMPLITUDE seed=$SEED cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_corner4x4_seed${SEED}.log

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
cap     = float('$TARGET_NORM')
trigger = '$TRIGGER'
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_corner4x4_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_corner4x4_seed{seed}_summary.txt'

LOG = open(LOG_PATH).read()

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
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

lines = [f'=== {trigger} probe: seed={seed} cap={cap} probe=gan_defense ===', '']
if pairs:
    fc, fa = pairs[-1]
    f100_net = np.mean([max(0, a-CTRL) for _,a in pairs[-100:]])
    lines += [f'  Final:    clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              f'  F100 mean net_asr: {f100_net:.3f}',
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
        lines += ['', f'  Detections: {cor_def}/{tot_atk} adv rounds = {det_rate:.3f}']

# Overlap stats
overlaps = re.findall(r'\[OVERLAP\] ep=(\d+) cos_adv_ben=([+-]?[\d.eE+-]+)', LOG)
if overlaps:
    ov_vals = [float(x[1]) for x in overlaps]
    lines += ['',
              f'  Overlap cos(adv,ben_mean): n={len(ov_vals)}',
              f'    mean={np.mean(ov_vals):.4f}  std={np.std(ov_vals):.4f}',
              f'    p5={np.percentile(ov_vals,5):.4f}  p50={np.percentile(ov_vals,50):.4f}  p95={np.percentile(ov_vals,95):.4f}']
else:
    lines += ['', '  [No OVERLAP rows found]']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
