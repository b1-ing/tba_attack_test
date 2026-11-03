#!/bin/bash
#SBATCH --job-name=step5_dtrack
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# D(t) accumulation tracking experiment.
#
# Fixes the ep=1 direction bug in accum_track/ortho_floor by capturing
# v = Delta_poison - Delta_clean (paired train) at ep>=50 (post-differentiation).
#
# Per round logs:
#   [V_CAPTURE]  ep, v_norm_raw, cos(adv_clipped, v)    -- one-time at ep>=50
#   [D_TRACK]    ep, D(t), adv_push, ben_erosion         -- every round after capture
#   [NORM_DIST]  ep, p5/p50/p95 of all client norms      -- every round
#
# 5 caps: {0.012, 0.030, 0.057, 0.100, 0.500}
# Slow caps (arms 1-2): 2400 rounds.  Fast caps (arms 3-5): 600 rounds.
# All FedAvg (defense=None) to isolate accumulation mechanics.

CAPS=(dummy 0.012 0.030 0.057 0.100 0.500)
ROUNDS_ARR=(dummy 2400 2400 600 600 600)
TARGET_NORM=${CAPS[$SLURM_ARRAY_TASK_ID]}
ROUNDS=${ROUNDS_ARR[$SLURM_ARRAY_TASK_ID]}
ARM_TAG="cap${TARGET_NORM/./p}"

SEED=1
V_CAPTURE_ROUND=50   # first round >= this where adversary present → capture v

echo "step5_dtrack job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "cap=$TARGET_NORM rounds=$ROUNDS arm_tag=$ARM_TAG"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_dtrack_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3/2024.10 python3.12 not found on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch algorithm.py: normcap + D(t) tracking ───────────────────────────
export SCRATCH_DIR TARGET_NORM V_CAPTURE_ROUND
$PYTHON -I - <<'PATCHEOF'
import os, sys

scratch          = os.environ['SCRATCH_DIR']
cap              = float(os.environ['TARGET_NORM'])
v_capture_round  = int(os.environ['V_CAPTURE_ROUND'])
alg              = f'{scratch}/FilterFL/algorithm.py'

with open(alg) as f:
    src = f.read()

# 1. Module-level state dict (before def FedAvg)
module_state = f'''
_DTRACK = {{'v': None, 'w0': None, 'capture_done': False}}
_V_CAPTURE_ROUND = {v_capture_round}
'''
fedavg_marker = 'def FedAvg(handle):'
if fedavg_marker not in src:
    print('ERROR: def FedAvg not found'); sys.exit(1)
src = src.replace(fedavg_marker, module_state + fedavg_marker, 1)

# 2. Pre-aggregation patch: combined normcap + D(t) tracking
pre_patch = f'''
        # ── [normcap + D(t) tracking] ────────────────────────────────────────
        _lnc_target = {cap}
        _dt_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        # Single loop: compute deltas, apply normcap, classify clients.
        # Save uncapped adversary delta separately — needed for v capture
        # (normcap shrinks small-cap deltas to near-zero, making d_poison-d_clean ≈ -d_clean).
        _dt_deltas       = []   # clipped deltas (used for D(t) push/erosion)
        _dt_is_adv       = []
        _dt_norms        = []
        _dt_adv_uncapped = None   # full-amplitude adversary delta before normcap
        for _dt_i, _dt_name in enumerate(agent_name_keys):
            _d = torch.cat([(w_locals[_dt_i][k].float() - ori_weight[k].float()).cpu().ravel()
                            for k in _dt_keys])
            _is_adv = (_dt_name in adversarial_name_keys)
            _dt_is_adv.append(_is_adv)
            if _is_adv:
                _dt_adv_uncapped = _d.clone()   # save before clipping
                _raw_norm = _d.norm().item()
                if _raw_norm > _lnc_target:
                    _scale = _lnc_target / _raw_norm
                    for k in _dt_keys:
                        w_locals[_dt_i][k] = (
                            ori_weight[k] +
                            (w_locals[_dt_i][k] - ori_weight[k]) * _scale
                        ).to(w_locals[_dt_i][k].dtype)
                    _d = _d * _scale
            _dt_deltas.append(_d)
            _dt_norms.append(_d.norm().item())

        _dt_adv_idxs = [i for i, a in enumerate(_dt_is_adv) if a]
        _dt_ben_idxs = [i for i, a in enumerate(_dt_is_adv) if not a]

        # Store w_0 (first round only, on CPU)
        if _DTRACK['w0'] is None:
            _DTRACK['w0'] = {{k: ori_weight[k].clone().cpu().float() for k in _dt_keys}}

        # One-time v capture via paired train at V_CAPTURE_ROUND.
        # Uses UNCAPPED d_poison so the trigger-specific component isn't dwarfed by -d_clean.
        # Assumes trigger injection is on-the-fly (poison=False gives clean delta).
        if (not _DTRACK['capture_done']
                and epoch >= _V_CAPTURE_ROUND
                and _dt_adv_idxs
                and _dt_adv_uncapped is not None):
            _adv_client = adversarial_name_keys[0]
            _model_cl   = copy.deepcopy(handle.model)
            _w_clean    = train.standard_train(
                epoch, handle.clients_data_num[_adv_client], _adv_client,
                handle.params, _model_cl.to(device),
                handle.train_data[_adv_client], handle, poison=False)
            _d_clean = torch.cat([(_w_clean[k].float() - ori_weight[k].float()).cpu().ravel()
                                  for k in _dt_keys])
            _v_raw  = _dt_adv_uncapped - _d_clean   # trigger-only direction (full amplitude)
            _v_norm = _v_raw.norm().item()
            if _v_norm > 1e-6:
                _DTRACK['v']            = (_v_raw / _v_norm).clone()
                _DTRACK['capture_done'] = True
                _adv_clipped = _dt_deltas[_dt_adv_idxs[0]]
                _cos_adv_v   = (_adv_clipped / (_adv_clipped.norm() + 1e-12)).dot(_DTRACK['v']).item()
                logger.info(
                    f'[V_CAPTURE] ep={{epoch}} v_norm_raw={{_v_norm:.5f}} '
                    f'adv_uncapped_norm={{_dt_adv_uncapped.norm():.5f}} '
                    f'clean_norm={{_d_clean.norm():.5f}} '
                    f'adv_clipped_norm={{_adv_clipped.norm():.5f}} '
                    f'cos_clipped_v={{_cos_adv_v:.4f}}'
                )
            else:
                logger.info(f'[V_CAPTURE_FAIL] ep={{epoch}} v_norm_raw={{_v_norm:.8f}} (poison~=clean)')

        # Per-round D(t), attacker push, benign erosion
        if _DTRACK['v'] is not None:
            _v  = _DTRACK['v']
            # Fix: move w0 tensors to same device as ori_weight before subtracting
            _wt = torch.cat([(ori_weight[k].float() - _DTRACK['w0'][k].to(ori_weight[k].device)).ravel()
                             for k in _dt_keys])
            _Dt = _wt.dot(_v.to(_wt.device)).item()
            _v_cpu = _v.cpu()
            _adv_push = (
                torch.stack([_dt_deltas[i] for i in _dt_adv_idxs]).sum(0).dot(_v_cpu).item() / lens
                if _dt_adv_idxs else 0.0
            )
            _ben_erosion = (
                torch.stack([_dt_deltas[i] for i in _dt_ben_idxs]).mean(0).dot(_v_cpu).item()
                * len(_dt_ben_idxs) / lens
                if _dt_ben_idxs else 0.0
            )
            logger.info(
                f'[D_TRACK] ep={{epoch}} D={{_Dt:.6f}} '
                f'adv_push={{_adv_push:.6f}} ben_erosion={{_ben_erosion:.6f}} '
                f'n_adv={{len(_dt_adv_idxs)}} n_ben={{len(_dt_ben_idxs)}}'
            )

        # Norm distribution across all clients this round
        _norms_t = torch.tensor(_dt_norms)
        logger.info(
            f'[NORM_DIST] ep={{epoch}} '
            f'p5={{float(_norms_t.quantile(0.05)):.5f}} '
            f'p50={{float(_norms_t.quantile(0.50)):.5f}} '
            f'p95={{float(_norms_t.quantile(0.95)):.5f}} '
            f'n_clients={{len(_dt_norms)}}'
        )
        # ── end normcap + D(t) tracking ──────────────────────────────────────
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
    print(f'dtrack patch: compiles OK (cap={cap}, v_capture_round={v_capture_round})')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
[[ $? -ne 0 ]] && { echo "Patch failed"; exit 1; }

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
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok: FedAvg cap=$TARGET_NORM rounds=$ROUNDS seed=$SEED', flush=True)
PYEOF

LOG=$OUT_DIR/step5_dtrack_${ARM_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense None 2>&1 | tee $LOG

echo "Training done."

# ── Post-processing ────────────────────────────────────────────────────────────
$PYTHON -I - <<PYEOF2
import re, sys, math
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

cap      = float('$TARGET_NORM')
arm_tag  = '$ARM_TAG'
rounds   = int('$ROUNDS')
CTRL     = 0.019
STABLE_K = 10   # rounds that must stay above 0.30 after first crossing to count as stable
STABLE_DROP = 0.30

log_path = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_dtrack_{arm_tag}.log'
out_path = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_dtrack_{arm_tag}_summary.txt'

LOG = open(log_path).read()

lines = [f'=== D(t) accumulation: cap={cap} rounds={rounds} seed=1 ===', '']

# ── V_CAPTURE ──
vc = re.search(
    r'\[V_CAPTURE\] ep=(\d+) v_norm_raw=([\d.eE+\-]+) adv_norm=([\d.eE+\-]+) '
    r'clean_norm=([\d.eE+\-]+) cos_adv_v=([\-\d.eE+\-]+)', LOG)
if vc:
    ep_vc, v_raw, adv_n, cln_n, cos = vc.groups()
    lines += [
        f'  v captured at ep={ep_vc}:',
        f'    v_norm_raw={float(v_raw):.5f}  adv_norm={float(adv_n):.5f}  clean_norm={float(cln_n):.5f}',
        f'    cos(adv_clipped, v) = {float(cos):.4f}  (1.0 = trigger-only dominates adv delta)',
        '']
elif re.search(r'\[V_CAPTURE_FAIL\]', LOG):
    lines.append('  [V_CAPTURE_FAIL: adv delta ~ clean delta; trigger may not have differentiated yet]')
    lines.append('')
else:
    lines.append('  [V_CAPTURE not found in log]')
    lines.append('')

# ── D(t) trajectory ──
dt_ms = re.findall(
    r'\[D_TRACK\] ep=(\d+) D=([\-\d.eE+\-]+) adv_push=([\-\d.eE+\-]+) ben_erosion=([\-\d.eE+\-]+)',
    LOG)
if dt_ms:
    eps       = [int(m[0])    for m in dt_ms]
    Ds        = [float(m[1])  for m in dt_ms]
    pushes    = [float(m[2])  for m in dt_ms]
    erosions  = [float(m[3])  for m in dt_ms]

    lines += ['  D(t) trajectory (every 200 rounds):']
    for i in range(0, len(eps), 200):
        lines.append(f'    ep={eps[i]:5d}  D={Ds[i]:+.5f}  push={pushes[i]:+.6f}  erosion={erosions[i]:+.6f}')
    if len(eps) > 1:
        lines.append(f'    ep={eps[-1]:5d}  D={Ds[-1]:+.5f}  push={pushes[-1]:+.6f}  erosion={erosions[-1]:+.6f}')
    lines.append('')

    # Erosion trend: compare early vs late (post ep=200)
    late_erosions = [e for ep, e in zip(eps, erosions) if ep > 200]
    if late_erosions:
        lines += [
            f'  Benign erosion (post ep=200): mean={np.mean(late_erosions):+.6f}  '
            f'std={np.std(late_erosions):.6f}',
            f'  (negative = eroding D; ~0 = erosion converged → accumulation limited only by patience)',
            '']
else:
    lines.append('  [No D_TRACK lines — v_capture may have failed]')
    lines.append('')

# ── ASR trajectory + stable implantation ──
evals = re.findall(r'(False|True),\s*([\d.]+)', LOG)
pairs = []
i = 0
while i + 1 < len(evals):
    if evals[i][0] == 'False' and evals[i+1][0] == 'True':
        pairs.append((float(evals[i][1]) / 100, float(evals[i+1][1]) / 100))
    i += 2

if pairs:
    net_asrs = [max(0, a - CTRL) for c, a in pairs]
    # Stable implantation: first crossing where next STABLE_K rounds don't drop below STABLE_DROP
    stable_impl = None
    for idx, net_asr in enumerate(net_asrs):
        if net_asr > 0.50:
            future = net_asrs[idx+1 : idx+1+STABLE_K]
            if len(future) == STABLE_K and all(x > STABLE_DROP for x in future):
                stable_impl = idx + 1
                break
            elif len(future) < STABLE_K:   # near end of run
                stable_impl = idx + 1
                break

    lines += [f'  ASR: final={pairs[-1][1]:.3f}  net_final={net_asrs[-1]:.3f}']
    lines += [f'  Stable implantation round (net_asr>0.50 sustained): {stable_impl}', '']

    # D* at stable implantation
    if stable_impl is not None and dt_ms:
        # Find D(t) values in the window [T_impl, T_impl+50]
        d_star_vals = [D for ep, D in zip(eps, Ds) if stable_impl <= ep <= stable_impl + 50]
        if d_star_vals:
            d_star = np.mean(d_star_vals)
            lines += [
                f'  D* (mean D(t) in [T_impl, T_impl+50]): {d_star:.6f}',
                f'  Rate estimate: c = {cap}  T_impl = {stable_impl}  c*T = {cap*stable_impl:.4f}',
                '']
    else:
        lines.append('  D* not computed (no stable implantation or D_TRACK missing)')
        lines.append('')
else:
    lines.append('  [No ASR eval pairs found]')
    lines.append('')

# ── Norm distribution (steady-state: post ep=200) ──
norm_ms = re.findall(
    r'\[NORM_DIST\] ep=(\d+) p5=([\d.eE+\-]+) p50=([\d.eE+\-]+) p95=([\d.eE+\-]+)',
    LOG)
if norm_ms:
    late = [(float(p5), float(p50), float(p95))
            for ep, p5, p50, p95 in norm_ms if int(ep) > 200]
    if late:
        p5s, p50s, p95s = zip(*late)
        lines += [
            f'  Norm distribution (steady-state, post ep=200, n={len(late)} rounds):',
            f'    p5  = {np.mean(p5s):.5f}  (detection ceiling: cap must be < this)',
            f'    p50 = {np.mean(p50s):.5f}',
            f'    p95 = {np.mean(p95s):.5f}',
            f'    cap = {cap}  →  cap < p5? {"YES (below detection ceiling)" if cap < np.mean(p5s) else "NO (detectable by norm filter)"}',
            '']
else:
    lines.append('  [No NORM_DIST lines found]')
    lines.append('')

out = '\n'.join(lines)
print(out)
with open(out_path, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {out_path}')
PYEOF2

echo "Done. Log: $LOG"
