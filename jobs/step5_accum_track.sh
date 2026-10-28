#!/bin/bash
#SBATCH --job-name=step5_accum
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Backdoor accumulation tracking.
# FedAvg (no defense), 600 rounds, 4 norm caps: 0.012/0.057/0.5/5.0.
#
# Key measurements per round:
#   BACKDOOR_PROJ ep=N proj=X    -- projection of (w_global - w_init) onto v_backdoor
#   [CLIENT_FLAGS] ...           -- adversary presence per round
#
# v_backdoor: unit vector in the direction of the adversary's full-amplitude
# local gradient (captured at first adversarial round, before norm clipping).
# w_init: global model weights at training start.
#
# Linearity check: if proj(t) ~ cap * t (linear in both norm and rounds),
# then small-norm + many-rounds = large-norm + few-rounds (scale-free accumulation).
# Weight->ASR mapping: compare proj(t) trajectory vs ASR(t) trajectory.
# If proj grows linearly but ASR has S-curve (slow start, plateau) -> nonlinear
# weight->ASR map (threshold + saturation). That explains "small still hits 98%":
# you only need to cross an ASR threshold, not match large attack's displacement.
#
# Arms (all FedAvg, seed=1, 600 rounds):
#   1: cap=0.012  (~1% benign norm, floor cap)
#   2: cap=0.057  (~5% benign norm, standard normcap)
#   3: cap=0.500  (~42% benign norm, louder)
#   4: cap=5.000  (~4x benign norm, very loud)

CAPS=(dummy 0.012 0.057 0.500 5.000)
CAP=${CAPS[$SLURM_ARRAY_TASK_ID]}
CAP_TAG=$(echo $CAP | sed 's/\./_/g' | sed 's/^0_/0p/' | sed 's/_0$//; s/_00$//')
# Produce clean tag: 0.012->0p012, 0.057->0p057, 0.500->0p500, 5.000->5p000
CAP_TAG="cap$(echo $CAP | sed 's/\./p/')"

SEED=1
ROUNDS=600

echo "step5_accum job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "FedAvg accumulation tracking: cap=$CAP tag=$CAP_TAG"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_accum_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

# Fail fast: anaconda3 not accessible on this node
[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3/2024.10 python3.12 not accessible on $(hostname)"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap + backdoor projection tracking ─────────────────────────────
# Two injection points:
#   (A) Before aggregation marker: norm cap + v_backdoor capture + CLIENT_FLAGS
#   (B) Before handle.model.load_state_dict(w_glob): projection computation
export SCRATCH_DIR CAP
$PYTHON -I - <<'PATCHEOF'
import os, sys

scratch  = os.environ['SCRATCH_DIR']
cap      = float(os.environ['CAP'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# ── (A) Pre-aggregation: normcap + v_backdoor init + CLIENT_FLAGS ─────────────
# _BACKTRACK dict holds v_backdoor (unit vec) and w0 (initial weights).
# Defined at module level so it persists across FedAvg() loop iterations.
# Using dict mutation (not rebind) so no 'global' declaration needed.
backtrack_init = '''
_BACKTRACK = {'v': None, 'w0': None}
'''

fedavg_marker = 'def FedAvg(handle):'
if fedavg_marker not in src:
    print('ERROR: FedAvg not found'); sys.exit(1)
src = src.replace(fedavg_marker, backtrack_init + fedavg_marker, 1)
print('_BACKTRACK init: OK')

pre_agg_patch = f'''
        # [NORMCAP+ACCUM] Cap adversarial delta; init v_backdoor at first adversarial round.
        _lnc_target = {cap}
        _lnc_keys   = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_delta_parts = [(w_locals[_lnc_i][k] - ori_weight[k]).float().cpu().ravel()
                                    for k in _lnc_keys]
                _lnc_d    = torch.cat(_lnc_delta_parts)
                _lnc_norm = _lnc_d.norm().item()
                # Capture v_backdoor from full-amplitude delta at first adversarial appearance
                if _BACKTRACK['v'] is None:
                    _BACKTRACK['v']  = (_lnc_d / _lnc_norm).clone().cpu()
                    _BACKTRACK['w0'] = torch.cat(
                        [ori_weight[k].float().cpu().ravel() for k in _lnc_keys]).clone()
                    logger.info(f"[ACCUM_INIT] ep={{epoch}} full_amp_norm={{_lnc_norm:.5f}} "
                                f"cap={cap:.5f}")
                # Apply norm cap
                if _lnc_norm > _lnc_target:
                    _lnc_scale = _lnc_target / _lnc_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_scale
                        ).to(w_locals[_lnc_i][k].dtype)
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]')
        # [END NORMCAP+ACCUM]
'''

agg_marker = '        # ── FLAME / FLTrust aggregation'
if agg_marker not in src:
    print('ERROR: aggregation marker not found'); sys.exit(1)
src = src.replace(agg_marker, pre_agg_patch + agg_marker, 1)
print('pre-agg patch (normcap + v_backdoor init): OK')

# ── (B) Post-aggregation: projection onto v_backdoor ─────────────────────────
post_agg_patch = '''
        # [BACKDOOR_PROJ] Project global model displacement onto v_backdoor direction.
        if _BACKTRACK['v'] is not None:
            _bt_keys = [k for k in w_glob if w_glob[k].dtype != torch.int64]
            _bt_curr = torch.cat([w_glob[k].float().cpu().ravel() for k in _bt_keys])
            _bt_proj = (_bt_curr - _BACKTRACK['w0']).dot(_BACKTRACK['v']).item()
            _bt_disp = (_bt_curr - _BACKTRACK['w0']).norm().item()
            logger.info(f"[BACKDOOR_PROJ] ep={epoch} proj={_bt_proj:.8f} "
                        f"displacement={_bt_disp:.6f}")
        # [END BACKDOOR_PROJ]
'''

post_marker = '        handle.model.load_state_dict(w_glob)'
if post_marker not in src:
    print('ERROR: post-aggregation marker not found'); sys.exit(1)
src = src.replace(post_marker, post_agg_patch + post_marker, 1)
print('post-agg patch (projection logging): OK')

with open(alg_path, 'w') as f:
    f.write(src)

import py_compile
try:
    py_compile.compile(alg_path, doraise=True)
    print(f'patched algorithm.py: compiles OK (cap={cap})')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML (FedAvg, no defense) ──────────────────────────────────────────────────
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
print('yaml ok: FedAvg defense=None cap=$CAP', flush=True)
PYEOF

LOG=$OUT_DIR/step5_accum_${CAP_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense None 2>&1 | tee $LOG

echo "Training done."

# ── Post-training analysis ─────────────────────────────────────────────────────
$PYTHON -I - <<PYEOF2
import re, sys
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

cap      = float('$CAP')
cap_tag  = '$CAP_TAG'
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_accum_{cap_tag}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_accum_{cap_tag}_summary.txt'
CTRL     = 0.019

LOG = open(LOG_PATH).read()
lines = [f'=== Accumulation tracking: FedAvg cap={cap} seed=1 rounds=$ROUNDS ===', '']

# ASR trajectory
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2
if pairs:
    fc, fa = pairs[-1]
    lines += [f'  Final: clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0,fa-CTRL):.3f}',
              '  ASR trajectory (every 50 evals):']
    for ep in list(range(0, len(pairs), 50)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
    impl = next((ep+1 for ep,(c,a) in enumerate(pairs) if a-CTRL > 0.5), None)
    lines.append(f'  Implantation round (net_asr>50%): {impl}')

# Projection trajectory
projs = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
         for m in re.finditer(
             r'\[BACKDOOR_PROJ\] ep=(\d+) proj=([\d.eE+\-]+) displacement=([\d.eE+\-]+)', LOG)]
if projs:
    eps, pvals, disps = zip(*projs)
    pv = np.array(pvals)
    dv = np.array(disps)
    lines += ['', f'  Backdoor projection (first/last/every 50):',
              f'    ep={eps[0]:4d}  proj={pv[0]:.6f}  disp={dv[0]:.6f}']
    for idx in range(0, len(projs), 50):
        lines.append(f'    ep={eps[idx]:4d}  proj={pv[idx]:.6f}  disp={dv[idx]:.6f}')
    lines.append(f'    ep={eps[-1]:4d}  proj={pv[-1]:.6f}  disp={dv[-1]:.6f}')
    lines += [f'  Final proj: {pv[-1]:.6f}  Final disp: {dv[-1]:.6f}',
              f'  Max proj: {np.max(pv):.6f}']
    # Linear check: does proj grow linearly with rounds?
    if len(pv) > 10:
        t = np.arange(len(pv))
        r = np.corrcoef(t, pv)[0,1]
        lines.append(f'  Proj-vs-round Pearson r={r:.4f} (1.0=linear, <0.9=sublinear/nonlinear)')

# Adversary appearances
adv_counts = [m.group(1).count('1')
              for m in re.finditer(r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\]', LOG)]
if adv_counts:
    ac = np.array(adv_counts)
    total_adv = int(np.sum(ac))
    dose = cap * total_adv / 10.0
    lines += ['', f'  Adv appearances: {total_adv}/{len(ac)} rounds',
              f'  Cumulative dose: {dose:.4f}  (cap={cap} * appearances/10)']

# ACCUM_INIT line
init_m = re.search(r'\[ACCUM_INIT\].*?full_amp_norm=([\d.eE+\-]+)', LOG)
if init_m:
    lines += ['', f'  Full-amplitude norm at init: {float(init_m.group(1)):.5f}',
              f'  Cap/full-amp ratio: {cap / float(init_m.group(1)):.5f}']

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
PYEOF2

echo "Done. Log: $LOG"
