#!/bin/bash
#SBATCH --job-name=step5_ortho
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Orthogonal-floor theory validation.
#
# Theory: floor_norm ~ f(N_clients, sigma_benign(v_orth))
# where v_orth = Gram-Schmidt(d_adv, mean_benign) — trigger-specific direction
# orthogonal to the benign gradient, removing the benign-alignment confound
# that made accum_track's projection metric useless.
#
# Measurement:
#   At first adversarial round:
#     d_adv     = adversary delta (full amplitude, before any cap)
#     mu_ben    = mean of benign client deltas
#     v_orth    = (d_adv - proj(d_adv, mu_ben_hat) * mu_ben_hat)
#     v_orth   /= ||v_orth||
#   Each subsequent round with v_orth known:
#     For each benign client i: p_i = dot(d_i, v_orth)
#     Log: std(p_i), mean(p_i), N_benign_clients
#
# Arms:
#   1: num_clients=10 (standard, N=10 selected per round)
#   2: num_clients=5  (half clients, N=5 selected per round)
#
# Prediction check (post-processing):
#   sigma_benign = mean std over rounds
#   If floor ~ N * sigma_benign: arm1 floor = 10 * sigma1
#                                arm2 floor = 5  * sigma2
#   If arm2 floor is ~half arm1 floor AND sigma2 ~ sigma1, the N-scaling holds.
#   Empirical floor = 0.012 (from normcap sweep; cap=0.012 fails, 0.057 succeeds).

N_CLIENTS_ARR=(dummy 10 5)
N_CLIENTS=${N_CLIENTS_ARR[$SLURM_ARRAY_TASK_ID]}
ARM_TAG="N${N_CLIENTS}"

SEED=1
ROUNDS=300   # 300 rounds is enough to estimate sigma stably
TARGET_NORM=0.5   # Use a large cap so adversary appears unclipped (full amplitude)

echo "step5_ortho job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "N_clients=$N_CLIENTS cap=$TARGET_NORM rounds=$ROUNDS"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_ortho_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3/2024.10 python3.12 not accessible on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap (large cap) + orthogonal v_orth computation + benign projection logging ──
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
cap     = float(os.environ['TARGET_NORM'])
alg     = f'{scratch}/FilterFL/algorithm.py'

with open(alg) as f:
    src = f.read()

# Module-level state dict (avoids global/nonlocal issues across loop iterations)
init_block = '''
_ORTHO = {'v': None}
'''

fedavg_marker = 'def FedAvg(handle):'
if fedavg_marker not in src:
    print('ERROR: FedAvg not found'); sys.exit(1)
src = src.replace(fedavg_marker, init_block + fedavg_marker, 1)

# Pre-aggregation patch: normcap + v_orth init + per-round benign projection logging
pre_patch = f'''
        # [ORTHO_FLOOR] Compute orthogonal backdoor direction and measure benign noise.
        _lnc_target = {cap}
        _of_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        # Collect client deltas (as flat float32 vectors, on CPU)
        _of_deltas = []
        _of_is_adv = []
        for _of_i, _of_name in enumerate(agent_name_keys):
            _d = torch.cat([(w_locals[_of_i][k] - ori_weight[k]).float().cpu().ravel()
                            for k in _of_keys])
            _of_deltas.append(_d)
            _is_adv = (_of_name in adversarial_name_keys)
            _of_is_adv.append(_is_adv)
            # Apply normcap to adversary
            if _is_adv:
                _norm = _d.norm().item()
                if _norm > _lnc_target:
                    _scale = _lnc_target / _norm
                    for k in _of_keys:
                        w_locals[_of_i][k] = (
                            ori_weight[k] +
                            (w_locals[_of_i][k] - ori_weight[k]) * _scale
                        ).to(w_locals[_of_i][k].dtype)

        _of_adv_idxs = [i for i, a in enumerate(_of_is_adv) if a]
        _of_ben_idxs = [i for i, a in enumerate(_of_is_adv) if not a]

        if _ORTHO['v'] is None and _of_adv_idxs:
            # First adversarial round: compute v_orth via Gram-Schmidt
            _d_adv = _of_deltas[_of_adv_idxs[0]]  # full-amplitude adversary delta
            _d_adv_norm = _d_adv.norm().item()
            if _of_ben_idxs:
                _mu_ben = torch.stack([_of_deltas[i] for i in _of_ben_idxs]).mean(0)
                _mu_ben_norm = _mu_ben.norm().item()
                _mu_ben_hat = _mu_ben / (_mu_ben_norm + 1e-12)
                # Gram-Schmidt: remove benign component from d_adv
                _proj_coeff = _d_adv.dot(_mu_ben_hat).item()
                _d_orth = _d_adv - _proj_coeff * _mu_ben_hat
                _d_orth_norm = _d_orth.norm().item()
                _cos_theta = (_d_adv.dot(_mu_ben_hat) / (_d_adv_norm + 1e-12)).item()
                if _d_orth_norm > 1e-6:
                    _ORTHO['v'] = (_d_orth / _d_orth_norm).clone()
                    logger.info(f"[ORTHO_INIT] ep={{epoch}} adv_norm={{_d_adv_norm:.5f}} "
                                f"ben_mean_norm={{_mu_ben_norm:.5f}} cos_adv_benMean={{_cos_theta:.4f}} "
                                f"orth_norm={{_d_orth_norm:.5f}} n_ben={{len(_of_ben_idxs)}}")
                else:
                    logger.info(f"[ORTHO_INIT_FAIL] ep={{epoch}} orth_norm too small; d_adv parallel to mean_ben")
            else:
                logger.info(f"[ORTHO_INIT_FAIL] ep={{epoch}} no benign clients this round")

        # Per-round: project benign deltas onto v_orth, log std + mean
        if _ORTHO['v'] is not None and _of_ben_idxs:
            _v = _ORTHO['v']
            _projs = [_of_deltas[i].dot(_v).item() for i in _of_ben_idxs]
            _p_arr = torch.tensor(_projs)
            _p_std  = _p_arr.std().item()
            _p_mean = _p_arr.mean().item()
            logger.info(f"[BENIGN_V_PROJ] ep={{epoch}} n={{len(_of_ben_idxs)}} "
                        f"std={{_p_std:.6f}} mean={{_p_mean:.6f}} "
                        f"vals={{','.join(f'{{p:.5f}}' for p in _projs)}}")
        # [END ORTHO_FLOOR]
'''

agg_marker = '        # ── FLAME / FLTrust aggregation'
if agg_marker not in src:
    print('ERROR: aggregation marker not found'); sys.exit(1)
src = src.replace(agg_marker, pre_patch + agg_marker, 1)

with open(alg, 'w') as f:
    f.write(src)

import py_compile
try:
    py_compile.compile(alg, doraise=True)
    print(f'ortho_floor patch: compiles OK (cap={cap})')
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
p['num_clients']                  = $N_CLIENTS
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
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok: FedAvg N_clients=$N_CLIENTS cap=$TARGET_NORM rounds=$ROUNDS', flush=True)
PYEOF

LOG=$OUT_DIR/step5_ortho_${ARM_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense None 2>&1 | tee $LOG

echo "Training done."

# ── Post-processing: compute sigma_benign(v_orth) and floor prediction ─────────
$PYTHON -I - <<PYEOF2
import re, sys, math
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

arm_tag  = '$ARM_TAG'
n_cl     = int('$N_CLIENTS')
log_path = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ortho_{arm_tag}.log'
out_path = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ortho_{arm_tag}_summary.txt'
EMP_FLOOR = 0.012   # empirical floor from normcap sweep

LOG = open(log_path).read()

lines = [f'=== Orthogonal floor: N_clients={n_cl} cap=$TARGET_NORM rounds=$ROUNDS ===', '']

# ORTHO_INIT line
init_m = re.search(
    r'\[ORTHO_INIT\] ep=(\d+) adv_norm=([\d.eE+\-]+) ben_mean_norm=([\d.eE+\-]+) '
    r'cos_adv_benMean=([\d.eE+\-]+) orth_norm=([\d.eE+\-]+) n_ben=(\d+)', LOG)
if init_m:
    ep, adv_n, ben_n, cos_th, orth_n, n_ben = init_m.groups()
    lines += [
        f'  v_orth init at ep={ep}:',
        f'    adv_norm={float(adv_n):.4f}  ben_mean_norm={float(ben_n):.4f}',
        f'    cos(d_adv, mu_ben)={float(cos_th):.4f}  (0=orthogonal, 1=aligned)',
        f'    orth_norm={float(orth_n):.4f}  (fraction of adv retained: {float(orth_n)/float(adv_n):.4f})',
        f'    n_benign_clients={n_ben}',
        '']
else:
    lines.append('  [ORTHO_INIT not found — patch may have failed]')

# BENIGN_V_PROJ: per-round std of benign projections
proj_ms = re.findall(
    r'\[BENIGN_V_PROJ\] ep=(\d+) n=(\d+) std=([\d.eE+\-]+) mean=([\d.eE+\-]+)', LOG)
if proj_ms:
    eps   = [int(m[0]) for m in proj_ms]
    ns    = [int(m[1]) for m in proj_ms]
    stds  = [float(m[2]) for m in proj_ms]
    means = [float(m[3]) for m in proj_ms]
    # Skip warmup (first 20 rounds)
    warmup = 20
    post = [(s, mu) for ep, s, mu in zip(eps, stds, means) if ep > warmup]
    if post:
        post_stds = [x[0] for x in post]
        post_means = [x[1] for x in post]
        sigma = np.mean(post_stds)
        sigma_std = np.std(post_stds)
        mean_proj = np.mean(post_means)
        lines += [
            f'  Benign projection onto v_orth (rounds >{warmup}):',
            f'    sigma_benign(v_orth) = {sigma:.6f}  ± {sigma_std:.6f}',
            f'    mean_benign(v_orth)  = {mean_proj:.6f}  (should be ~0 by construction)',
            f'    n_rounds measured    = {len(post_stds)}',
            '']
        # Floor predictions
        lines.append('  Floor predictions:')
        for label, factor in [
            ('cap > sigma (per-round SNR >= 1)', sigma),
            ('cap > N * sigma (N=n_clients)', n_cl * sigma),
            ('cap > sqrt(N) * sigma', math.sqrt(n_cl) * sigma),
            ('cap > sigma / sqrt(rounds)', sigma / math.sqrt(300)),
        ]:
            marker = ' <-- MATCHES' if abs(math.log10(factor) - math.log10(EMP_FLOOR)) < 0.5 else ''
            lines.append(f'    {label}: {factor:.6f}{marker}')
        lines += ['',
            f'  Empirical floor (from normcap sweep): {EMP_FLOOR}',
            f'  Ratio empirical/sigma: {EMP_FLOOR/sigma:.3f}']
else:
    lines.append('  [No BENIGN_V_PROJ lines found]')

out = '\n'.join(lines)
print(out)
with open(out_path, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {out_path}')
PYEOF2

echo "Done. Log: $LOG"
