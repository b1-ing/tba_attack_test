#!/bin/bash
#SBATCH --job-name=step5_fang
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Fang et al. (USENIX 2020) adaptive attack, adapted for backdoor.
#
# Original Fang attack: max lambda s.t. w' = w_global - lambda*s is Krum-selected,
# where s = sign of update direction (untargeted error-max).
#
# Adaptation for backdoor (partial-knowledge, c=1):
#   s_hat = (w_poisoned - w_global) / ||w_poisoned - w_global||  (poisoned direction)
#   Binary search max lambda s.t. w' = w_global + lambda*s_hat is Krum-selected
#   (in Krum arm) or just submit max lambda bounded by benign norm statistics.
#
# For TrimmedMean arm: Fang's per-coordinate extremal placement (partial-knowledge
# Gaussian variant: push each coordinate to mean ± 3-4*sigma in attack direction).
#
# Threat model: grey-box — attacker sees all w_locals at aggregation time.
#   Contrast with normcap (black-box: no knowledge of others' updates).
#   Same level as Neurotoxin.
#
# Arms:
#   1: FilterFL    (ours)      — Fang backdoor adaptive attack
#   2: FLAME                   — Fang backdoor adaptive attack
#   3: FLTrust                 — Fang backdoor adaptive attack
#   4: Multi-Krum (m=5)        — Fang Krum attack (designed target)
#   5: TrimmedMean             — Fang Trim attack (designed target)
#
# Config: n=10, f=1, k=7, m=5, alpha=0.5, checker, seed=1, 600 rounds.
# Baseline comparison: normcap at 0.057 (black-box).

DEFENSES=(dummy ours flame fltrust krum trimmedmean)
DEF_TAGS=(dummy filterfl flame fltrust krum trimmedmean)
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

KRUM_M=5
SEED=1
MAX_NORM_SEARCH=5.0   # upper bound for binary search (well above benign ~1.2)

echo "step5_fang job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Fang-adapted attack: defense=$DEFENSE seed=$SEED"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_fang_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: Fang adaptive attack + Krum/TrimmedMean impls + CLIENT_FLAGS ─────────
export SCRATCH_DIR KRUM_M MAX_NORM_SEARCH DEF_TAG
python3 -I - <<'PATCHEOF'
import os, sys
scratch  = os.environ['SCRATCH_DIR']
km       = int(os.environ['KRUM_M'])
mns      = float(os.environ['MAX_NORM_SEARCH'])
def_tag  = os.environ['DEF_TAG']
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# ── Krum aggregation function ─────────────────────────────────────────────────
krum_fn = f'''
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
    logger.info(f\'[KRUM] ep={{epoch}} m={{m}} scores={{[round(s,2) for s in scores]}} selected={{selected}}\')
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
src = src.replace(insert_before, krum_fn + insert_before, 1)

# ── Fang adaptive attack patch ────────────────────────────────────────────────
# Strategy:
#   - For krum/trimmedmean arms: full Fang adaptive (binary search / per-coord extremal)
#   - For filterfl/flame/fltrust arms: Fang-direction with Krum-adaptive norm
#     (max lambda that would be selected under Krum — tests cross-defense behavior)
#
# All arms: after local training, compute attack direction from poisoned delta.
# Krum arm: binary search max lambda for Krum selection.
# Trim arm: per-coordinate extremal placement (partial-knowledge Gaussian variant).
# Other arms: Krum-adaptive lambda (binary search against krum metric, submit to different defense).

fang_patch = f'''
        # [FANG] Fang et al. (USENIX 2020) adaptive attack, adapted for backdoor.
        # Grey-box: attacker observes all w_locals at aggregation time.
        _fang_def   = '{def_tag}'
        _fang_keys  = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        _fang_n     = len(w_locals)
        _fang_kf    = handle.params.get('krum_f', 1)
        _fang_km    = handle.params.get('krum_m', 1)
        _fang_kk    = max(1, _fang_n - _fang_kf - 2)

        def _fang_krum_score(vecs, target_idx):
            n = len(vecs)
            dists_to = [(vecs[target_idx] - vecs[j]).pow(2).sum().item()
                        for j in range(n) if j != target_idx]
            return sum(sorted(dists_to)[:_fang_kk])

        for _fi, _fname in enumerate(agent_name_keys):
            if _fname not in adversarial_name_keys:
                continue

            # Poisoned delta and direction
            _f_delta = torch.cat([(w_locals[_fi][k] - ori_weight[k]).float().cpu().ravel()
                                   for k in _fang_keys])
            _f_norm  = _f_delta.norm().item()
            if _f_norm < 1e-10:
                continue
            _f_dhat  = _f_delta / _f_norm   # unit vector in poisoned direction

            # Benign delta vectors (all non-adversarial)
            _ben_vecs = [torch.cat([(w_locals[j][k] - ori_weight[k]).float().cpu().ravel()
                                     for k in _fang_keys])
                         for j in range(_fang_n) if agent_name_keys[j] not in adversarial_name_keys]

            if _fang_def == 'trimmedmean':
                # Fang Trim attack (partial-knowledge Gaussian variant):
                # Per coordinate: push adversarial value 3-4 sigma outside benign range,
                # in the direction that hurts the trimmed mean.
                # Estimate per-coord mean/std from benign deltas.
                _ben_mat  = torch.stack(_ben_vecs, dim=0)   # (n_ben, d)
                _ben_mu   = _ben_mat.mean(dim=0)
                _ben_sig  = _ben_mat.std(dim=0).clamp(min=1e-6)
                # Attack direction: toward our poisoned objective
                # s_j = sign(f_delta_j): push coordinate further in that direction
                _s        = _f_dhat.sign()
                # Place at mu + 3.5*sigma in s direction (just outside trim range)
                _trim_delta = _ben_mu + 3.5 * _ben_sig * _s
                _offset = 0
                for k in _fang_keys:
                    numel = ori_weight[k].numel()
                    chunk = _trim_delta[_offset:_offset+numel].view(ori_weight[k].shape)
                    w_locals[_fi][k] = (ori_weight[k] +
                                        chunk.to(device=ori_weight[k].device,
                                                 dtype=ori_weight[k].dtype))
                    _offset += numel
                _final_norm = _trim_delta.norm().item()
                logger.info(f\'[FANG] ep={{epoch}} adv={{_fname}} mode=trim \' +
                            f\'norm={{_final_norm:.5f}}\')

            else:
                # Krum-adaptive binary search (used for krum arm AND cross-defense arms).
                # Find max lambda s.t. attacker Krum score <= benign min Krum score.
                #
                # Optimisation: precompute dot products and squared norms of benign
                # deltas vs attack direction once per round. Then for any lambda:
                #   dist²(lambda*d_hat, ben_j) = lambda² - 2*lambda*dot_j + norm_sq_j
                # Binary search becomes pure scalar arithmetic — O(n) not O(n*d).
                #
                # Only benign Krum score computation requires O(n²d) work (done once).
                _ben_vecs_list = [torch.cat([(w_locals[j][k] - ori_weight[k]).float().cpu().ravel()
                                              for k in _fang_keys])
                                   for j in range(_fang_n)
                                   if agent_name_keys[j] not in adversarial_name_keys]
                _n_ben = len(_ben_vecs_list)

                # Pairwise benign distances for ben_min (done once, O(n_ben² * d))
                _ben_pw = [[0.0]*_n_ben for _ in range(_n_ben)]
                for _i in range(_n_ben):
                    for _j in range(_i+1, _n_ben):
                        _d = (_ben_vecs_list[_i] - _ben_vecs_list[_j]).pow(2).sum().item()
                        _ben_pw[_i][_j] = _ben_pw[_j][_i] = _d
                _ben_scores = []
                for _i in range(_n_ben):
                    _row = sorted(_ben_pw[_i])
                    _ben_scores.append(sum(_row[:_fang_kk]))
                _ben_min = min(_ben_scores) if _ben_scores else 1e9

                # Precompute dot products and norms for binary search (O(n_ben * d))
                _dots   = torch.tensor([(_f_dhat @ _ben_vecs_list[j]).item()
                                         for j in range(_n_ben)])
                _norms2 = torch.tensor([_ben_vecs_list[j].pow(2).sum().item()
                                         for j in range(_n_ben)])

                def _lam_krum_score(lam):
                    # dist²(lam*d_hat, ben_j) = lam² - 2*lam*dot_j + norm_sq_j
                    dists_sq = lam*lam - 2*lam*_dots + _norms2  # shape (n_ben,)
                    return float(dists_sq.sort().values[:_fang_kk].sum())

                # Binary search: max lambda achieving krum_score <= ben_min
                _lo, _hi = 0.0, {mns}
                _best_lam = 0.0
                for _ in range(25):
                    _mid = (_lo + _hi) / 2.0
                    if _lam_krum_score(_mid) <= _ben_min:
                        _best_lam = _mid
                        _lo = _mid
                    else:
                        _hi = _mid
                    if _hi - _lo < 1e-5:
                        break

                _final_delta = _best_lam * _f_dhat
                _final_norm  = _final_delta.norm().item()
                logger.info(f\'[FANG] ep={{epoch}} adv={{_fname}} mode=krum_adaptive \' +
                            f\'lambda={{_best_lam:.5f}} norm={{_final_norm:.5f}} \' +
                            f\'krum_score={{_lam_krum_score(_best_lam):.2f}} \' +
                            f\'ben_min_krum={{_ben_min:.2f}}\')

                _offset = 0
                for k in _fang_keys:
                    numel = ori_weight[k].numel()
                    chunk = _final_delta[_offset:_offset+numel].view(ori_weight[k].shape)
                    w_locals[_fi][k] = (ori_weight[k] +
                                        chunk.to(device=ori_weight[k].device,
                                                 dtype=ori_weight[k].dtype))
                    _offset += numel

        # CLIENT_FLAGS
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f\'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]\')
        # [END FANG]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, fang_patch + marker, 1)

# ── Aggregation dispatch: add krum + trimmedmean ──────────────────────────────
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

with open(alg_path, 'w') as f:
    f.write(src)
print(f'fang patch applied (def={def_tag}, krum_m={km})', flush=True)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ──────────────────────────────────────────────────────────────────────
python3 -I - <<PYEOF
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
print('yaml ok defense=$DEFENSE seed=$SEED', flush=True)
PYEOF

LOG=$OUT_DIR/step5_fang_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Post-training analysis ─────────────────────────────────────────────────────
python3 -I - <<PYEOF2
import re, numpy as np
log      = open('$LOG').read()
def_tag  = '$DEF_TAG'
out_file = '$OUT_DIR/step5_fang_${DEF_TAG}_summary.txt'
CTRL     = 0.019

lines = [f'=== Fang attack: defense={def_tag} seed=$SEED ===', '']

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
    lines.append('  Trajectory (every 100):')
    for ep in range(0, len(pairs), 100):
        c, a = pairs[ep]
        lines.append(f'    eval={ep+1:4d}  clean={c:.3f}  asr={a:.3f}  net_asr={max(0,a-CTRL):.3f}')
else:
    lines.append('  [no eval rows found]')

# Fang lambda trajectory
lam_vals = [float(m.group(1)) for m in
            re.finditer(r'lambda=([\d.eE+-]+)', log)]
if lam_vals:
    lv = np.array(lam_vals)
    lines += ['',
              f'  Fang adaptive lambda (norm submitted):',
              f'    n_rounds={len(lv)}  mean={np.mean(lv):.4f}  '
              f'p5={np.percentile(lv,5):.4f}  p95={np.percentile(lv,95):.4f}',
              f'    Baseline normcap: 0.057',
              f'    Ratio (fang/normcap): {np.mean(lv)/0.057:.1f}x']

# Krum selection (if krum arm)
krum_lines = re.findall(r'\[KRUM\].*', log)
if krum_lines:
    flags = {}
    for m in re.finditer(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]', log):
        ep   = int(m.group(1))
        mask = list(map(int, m.group(3).split(',')))
        flags[ep] = [i for i,v in enumerate(mask) if v==1]
    sel_count, total = 0, 0
    for m in re.finditer(r'\[KRUM\] ep=(\d+) m=\d+ scores=\[([^\]]+)\] selected=\[([^\]]+)\]', log):
        ep  = int(m.group(1))
        sel = list(map(int, m.group(3).split(',')))
        if ep in flags:
            for ai in flags[ep]:
                total += 1
                if ai in sel: sel_count += 1
    lines += ['', f'  Krum selection: {sel_count}/{total} ({100*sel_count/max(1,total):.0f}%)']

# FilterFL detection
det = len(re.findall(r'FEAT_ALIGN_DETECT', log))
fa  = len(re.findall(r'FEAT_ALIGN ep=', log))
if def_tag == 'filterfl' and fa > 0:
    lines += ['', f'  FilterFL detections: {det}/{fa} ({100*det/fa:.1f}%)']

out = '\n'.join(lines)
print(out)
with open(out_file, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {out_file}')
PYEOF2

echo "Done. Log: $LOG"
