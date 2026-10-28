#!/bin/bash
#SBATCH --job-name=step5_alie
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# A Little Is Enough (ALIE) — Baruch et al., NeurIPS 2019.
# Canonical low-norm FL attack from literature, run unchanged against 5 defenses.
#
# Algorithm (partial-knowledge, c adversaries per round):
#   1. Each adversary trains locally on poisoned data (checker trigger, 50% rate).
#   2. At aggregation: mu = coord-wise mean of adversarial deltas
#                      sigma = coord-wise std of adversarial deltas
#   3. All adversaries submit: delta_adv = mu + tau * sigma
#      (every adversary in the round submits the same crafted vector)
#   tau = 1.5 (ByzFL/Baruch default)
#
# Threat model: partial-knowledge grey-box — attacker knows only the c compromised
#   clients' own updates (no benign gradient access). With c>=2 per round, mu and
#   sigma are estimated from the compromised pool.
#
# Setup: poison_client_num=30 of 100 (30%), num_clients=10 per round.
#   E[adversaries per round] = 3. Rounds with <2 adversaries fall back to
#   unmodified poisoned update (can't estimate sigma with <2 samples).
#
# Comparison point: normcap c=1, cap=0.057 (black-box, single attacker).
# Key questions:
#   - Does ALIE's output norm differ from 0.057?
#   - Does ALIE evade all 5 defenses?
#   - Does ALIE implant a backdoor (ASR)?
#
# Arms:
#   1: FilterFL (ours)
#   2: FLAME
#   3: FLTrust
#   4: Multi-Krum (m=5)
#   5: TrimmedMean

DEFENSES=(dummy ours flame fltrust krum trimmedmean)
DEF_TAGS=(dummy filterfl flame fltrust krum trimmedmean)
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

TAU=1.5
KRUM_M=5
SEED=1
ADV_POOL=30    # 30% of 100 participants are adversaries

echo "step5_alie job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "ALIE tau=$TAU: defense=$DEFENSE seed=$SEED adv_pool=$ADV_POOL/100"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_alie_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: ALIE + Krum/TrimmedMean impls + CLIENT_FLAGS ──────────────────────
export SCRATCH_DIR KRUM_M TAU DEF_TAG
python3 -I - <<'PATCHEOF'
import os, sys
scratch  = os.environ['SCRATCH_DIR']
km       = int(os.environ['KRUM_M'])
tau      = float(os.environ['TAU'])
def_tag  = os.environ['DEF_TAG']
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# ── Krum + TrimmedMean aggregation functions ──────────────────────────────────
agg_fns = f'''
def _krum_aggregate(w_locals, ori_weight, keys, params, logger, epoch, lens):
    n   = len(w_locals)
    f   = params.get('krum_f', 1)
    m   = params.get('krum_m', 1)
    k   = max(1, n - f - 2)
    eta = params.get('eta', 1.0)
    vecs = [torch.cat([(w_locals[i][key] - ori_weight[key]).float().cpu().ravel()
                        for key in keys])
            for i in range(n)]
    dists = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = (vecs[i] - vecs[j]).pow(2).sum().item()
            dists[i][j] = dists[j][i] = d
    scores = []
    for i in range(n):
        row = sorted(dists[i][:i] + dists[i][i+1:])
        scores.append(sum(row[:k]))
    order = sorted(range(n), key=lambda i: scores[i])
    selected = order[:m]
    logger.info(f"[KRUM] ep={{epoch}} m={{m}} scores={{[round(s,2) for s in scores]}} selected={{selected}}")
    w_out = {{key: ori_weight[key].clone() for key in ori_weight}}
    for key in keys:
        delta = sum((w_locals[i][key] - ori_weight[key]).float() for i in selected) / m
        w_out[key] = (ori_weight[key] + eta * delta.to(ori_weight[key].dtype))
    for key in ori_weight:
        if ori_weight[key].dtype == torch.int64:
            w_out[key] = ori_weight[key].clone()
    return w_out

def _trimmedmean_aggregate(w_locals, ori_weight, keys, params, logger, epoch):
    n    = len(w_locals)
    beta = int(params.get('trimmedmean_beta', 1))
    eta  = params.get('eta', 1.0)
    w_out = {{key: ori_weight[key].clone() for key in ori_weight}}
    for key in keys:
        stacked = torch.stack([(w_locals[i][key] - ori_weight[key]).float()
                                for i in range(n)], dim=0)
        sorted_d, _ = stacked.sort(dim=0)
        trimmed     = sorted_d[beta:n-beta]
        delta       = trimmed.mean(dim=0)
        w_out[key]  = (ori_weight[key] + eta * delta.to(ori_weight[key].dtype))
    for key in ori_weight:
        if ori_weight[key].dtype == torch.int64:
            w_out[key] = ori_weight[key].clone()
    return w_out

'''

insert_before = 'def FedAvg(handle):'
if insert_before not in src:
    print('ERROR: FedAvg not found'); sys.exit(1)
src = src.replace(insert_before, agg_fns + insert_before, 1)
print('agg_fns injection: OK')

# ── ALIE patch (injected before aggregation marker) ───────────────────────────
alie_patch = f'''
        # [ALIE] A Little Is Enough (Baruch et al., NeurIPS 2019).
        # Partial-knowledge: mu/sigma estimated from compromised clients only.
        # All adversaries in the round submit the same crafted delta: mu + tau*sigma.
        _alie_tau  = {tau}
        _alie_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        # Collect adversarial indices in this round
        _adv_idxs = [i for i, n in enumerate(agent_name_keys)
                     if n in adversarial_name_keys]

        if len(_adv_idxs) >= 2:
            # Stack adversarial deltas: shape (c, d_flat)
            _adv_deltas = torch.stack([
                torch.cat([(w_locals[i][k] - ori_weight[k]).float().cpu().ravel()
                            for k in _alie_keys])
                for i in _adv_idxs
            ], dim=0)   # (c, d)
            _mu    = _adv_deltas.mean(dim=0)          # (d,)
            _sigma = _adv_deltas.std(dim=0)           # (d,)  sample std
            _alie_delta = _mu + _alie_tau * _sigma    # (d,)
            _alie_norm  = _alie_delta.norm().item()
            logger.info(f"[ALIE] ep={{epoch}} n_adv={{len(_adv_idxs)}} tau={{_alie_tau}} "
                        f"norm={{_alie_norm:.5f}} mu_norm={{_mu.norm().item():.5f}} "
                        f"sigma_norm={{_sigma.norm().item():.5f}}")
            # Write back: all adversaries submit the same ALIE vector
            for _ai in _adv_idxs:
                _offset = 0
                for k in _alie_keys:
                    numel  = ori_weight[k].numel()
                    chunk  = _alie_delta[_offset:_offset+numel].view(ori_weight[k].shape)
                    w_locals[_ai][k] = (ori_weight[k] +
                                        chunk.to(device=ori_weight[k].device,
                                                 dtype=ori_weight[k].dtype))
                    _offset += numel
        elif len(_adv_idxs) == 1:
            # Fallback with 1 adversary: submit own poisoned delta unmodified
            _ai = _adv_idxs[0]
            _d  = torch.cat([(w_locals[_ai][k] - ori_weight[k]).float().cpu().ravel()
                              for k in _alie_keys])
            logger.info(f"[ALIE] ep={{epoch}} n_adv=1 fallback norm={{_d.norm().item():.5f}}")
        # else: 0 adversaries in this round, nothing to do

        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f"[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]")
        # [END ALIE]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, alie_patch + marker, 1)
print('alie_patch injection: OK')

# ── Aggregation dispatch ──────────────────────────────────────────────────────
old_else = (
    "        if _def == 'fltrust':\n"
    "            w_glob = fltrust_aggregate(\n"
    "                {k: v.clone() for k, v in ori_weight.items()},\n"
    "                w_locals, _root_x, _root_y, handle.params, epoch, device, logger)\n"
    "        else:\n"
    "            w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},\n"
    "                                 w_locals, lens, beta, handle.params['eta'],\n"
    "                                 _def, handle.params)"
)
new_else = (
    "        if _def == 'fltrust':\n"
    "            w_glob = fltrust_aggregate(\n"
    "                {k: v.clone() for k, v in ori_weight.items()},\n"
    "                w_locals, _root_x, _root_y, handle.params, epoch, device, logger)\n"
    "        elif _def == 'krum':\n"
    "            w_glob = _krum_aggregate(w_locals, ori_weight, _keys, handle.params,\n"
    "                                     logger, epoch, lens)\n"
    "        elif _def == 'trimmedmean':\n"
    "            w_glob = _trimmedmean_aggregate(w_locals, ori_weight, _keys,\n"
    "                                            handle.params, logger, epoch)\n"
    "        else:\n"
    "            w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},\n"
    "                                 w_locals, lens, beta, handle.params['eta'],\n"
    "                                 _def, handle.params)"
)
if old_else not in src:
    print('ERROR: else block not found'); sys.exit(1)
src = src.replace(old_else, new_else, 1)
print('dispatch injection: OK')

with open(alg_path, 'w') as f:
    f.write(src)

import py_compile
try:
    py_compile.compile(alg_path, doraise=True)
    print('patched algorithm.py: compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ──────────────────────────────────────────────────────────────────────
python3 -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = $ADV_POOL   # 30 adversaries in pool
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = 600
p['seed']                         = $SEED
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['krum_f']                       = 1
p['krum_m']                       = $KRUM_M
p['trimmedmean_beta']             = 1
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
p['log_ben_s3']                   = ('$DEFENSE' == 'ours')
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok defense=$DEFENSE adv_pool=$ADV_POOL', flush=True)
PYEOF

LOG=$OUT_DIR/step5_alie_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Post-training analysis ─────────────────────────────────────────────────────
python3 -I - <<PYEOF2
import re, numpy as np
log      = open('$LOG').read()
def_tag  = '$DEF_TAG'
out_file = '$OUT_DIR/step5_alie_${DEF_TAG}_summary.txt'
CTRL     = 0.019

lines = [f'=== ALIE (tau=$TAU, adv_pool=$ADV_POOL/100): defense={def_tag} seed=$SEED ===', '']

# ASR
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', log)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2
if pairs:
    final_c, final_a = pairs[-1]
    lines.append(f'  Final: clean={final_c:.3f}  asr={final_a:.3f}  net_asr={max(0,final_a-CTRL):.3f}')
    lines.append(f'  Mean clean acc: {np.mean([c for c,a in pairs]):.3f}')
    lines.append('  Trajectory (every 100 evals):')
    for ep in list(range(0, len(pairs), 100)) + [len(pairs)-1]:
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
else:
    lines.append('  [no eval rows found]')

# ALIE norm trajectory
alie_norms = [float(m.group(1)) for m in
              re.finditer(r'\[ALIE\].*?norm=([\d.eE+-]+)', log)]
n1_counts  = len(re.findall(r'\[ALIE\].*?n_adv=1 fallback', log))
if alie_norms:
    an = np.array(alie_norms)
    lines += ['',
              f'  ALIE norm (rounds with >=2 adv): n={len(an)} fallback_rounds={n1_counts}',
              f'    mean={np.mean(an):.4f}  p5={np.percentile(an,5):.4f}  '
              f'p95={np.percentile(an,95):.4f}',
              f'    Normcap baseline (c=1): 0.057']

# Adversaries per round stats
adv_counts = [m.group(1).count('1')
              for m in re.finditer(r'\[CLIENT_FLAGS\].*?adv_mask=\[([^\]]+)\]', log)]
if adv_counts:
    ac = np.array(adv_counts)
    lines += ['',
              f'  Adv per round: mean={np.mean(ac):.2f}  min={np.min(ac)}  max={np.max(ac)}']

# FilterFL detection
det = len(re.findall(r'FEAT_ALIGN_DETECT', log))
fa  = len(re.findall(r'FEAT_ALIGN ep=', log))
if def_tag == 'filterfl' and fa > 0:
    lines += ['', f'  FilterFL detections: {det}/{fa} ({100*det/fa:.1f}%)']

# Krum selection
krum_lines = re.findall(r'\[KRUM\]', log)
if krum_lines:
    flags = {}
    for m in re.finditer(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]', log):
        ep   = int(m.group(1))
        mask = list(map(int, m.group(3).split(',')))
        flags[ep] = [i for i,v in enumerate(mask) if v==1]
    sel, tot = 0, 0
    for m in re.finditer(r'\[KRUM\] ep=(\d+) m=\d+ scores=\[([^\]]+)\] selected=\[([^\]]+)\]', log):
        ep  = int(m.group(1))
        sel_list = list(map(int, m.group(3).split(',')))
        if ep in flags:
            for ai in flags[ep]:
                tot += 1
                if ai in sel_list: sel += 1
    lines += ['', f'  Krum selection: {sel}/{tot} ({100*sel/max(1,tot):.0f}%)']

out = '\n'.join(lines)
print(out)
with open(out_file, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {out_file}')
PYEOF2

echo "Done. Log: $LOG"
