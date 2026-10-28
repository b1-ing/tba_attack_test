#!/bin/bash
#SBATCH --job-name=step5_xfam2
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Cross-family test: Krum and Trimmed Mean.
#
# PREDICTIONS (derived from defense criterion before running):
#   Krum (n=10, f=1, k=7):
#     - Attacker norm ~0.057, benign ~1.2-1.3.
#     - dist(attacker, benign_j) ≈ ||benign_j|| ≈ 1.2 (attacker is near-zero).
#     - dist(benign_i, benign_j) ≈ sqrt(2)*1.2*(1-cos_sim)^0.5 ≈ 1.4 (diverse non-IID).
#     - Krum score(attacker) = sum 7 nearest ≈ 7*1.44 = 10.1
#     - Krum score(benign_i) ≈ 6*2.0 + 1.44 = 13.4
#     - PREDICTION: attacker appears CENTRAL → Krum selects it → attack evades.
#   Trimmed Mean (beta=1, trim 1 from each end per coordinate):
#     - Attacker per-coordinate values are small (not outliers).
#     - Small values survive trimming (they're in the middle of the distribution).
#     - PREDICTION: attacker not trimmed → contributes at 1/(n-2) weight → attack evades.
#
# Both predictions: evades by being "invisible" (small norm / non-outlier per coordinate).
# Goal: confirm predictions hold empirically (predictive theory over 3 data points).
#
# Arms:
#   1: Krum         seed=1, cap=0.057
#   2: TrimmedMean  seed=1, cap=0.057
#
# Also logs: per-client Krum scores / trim status + CLIENT_FLAGS for distributional analysis.

TARGET_NORM=0.057
DEFENSES=(dummy krum trimmedmean)
DEF_TAGS=(dummy krum trimmedmean)

DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

echo "step5_xfam2 job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_xfam2_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: inject Krum/TrimmedMean + normcap + CLIENT_FLAGS ──────────────────
export SCRATCH_DIR TARGET_NORM DEFENSE
python3 -I - <<'PATCHEOF'
import os, sys
scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
defense  = os.environ['DEFENSE']
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# ── 1. Inject helper functions before FedAvg ─────────────────────────────────
krum_tm_fns = '''
def _krum_aggregate(w_locals, ori_weight, keys, params, logger, epoch, lens):
    """
    Krum / Multi-Krum aggregation.
    n clients, f expected adversaries → k = n - f - 2 nearest neighbors per client.
    Selects m clients (krum_m) with lowest distance scores.
    """
    n   = len(w_locals)
    f   = params.get('krum_f', 1)
    m   = params.get('krum_m', 1)
    k   = max(1, n - f - 2)
    eta = params.get('eta', 1.0)
    # Flatten deltas
    vecs = [torch.cat([(w_locals[i][key] - ori_weight[key]).float().cpu().ravel()
                        for key in keys])
            for i in range(n)]
    # Pairwise squared distances
    dists = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = (vecs[i] - vecs[j]).pow(2).sum().item()
            dists[i][j] = dists[j][i] = d
    # Krum scores
    scores = []
    for i in range(n):
        row = sorted(dists[i][:i] + dists[i][i+1:])
        scores.append(sum(row[:k]))
    order = sorted(range(n), key=lambda i: scores[i])
    selected = order[:m]
    logger.info(f\'[KRUM] ep={epoch} scores={[round(s,3) for s in scores]} \' +
                f\'selected={selected}\')
    # Average selected clients
    w_out = {key: ori_weight[key].clone() for key in ori_weight}
    for key in keys:
        delta = sum((w_locals[i][key] - ori_weight[key]).float() for i in selected) / m
        w_out[key] = (ori_weight[key] + eta * delta.to(ori_weight[key].dtype))
    for key in ori_weight:
        if ori_weight[key].dtype == torch.int64:
            w_out[key] = ori_weight[key].clone()
    return w_out


def _trimmedmean_aggregate(w_locals, ori_weight, keys, params, logger, epoch):
    """
    Coordinate-wise Trimmed Mean. Trims beta clients from each end per coordinate.
    """
    n    = len(w_locals)
    beta = params.get('trim_beta', 1)
    eta  = params.get('eta', 1.0)
    w_out = {key: ori_weight[key].clone() for key in ori_weight}
    for key in keys:
        stacked = torch.stack([(w_locals[i][key] - ori_weight[key]).float()
                                for i in range(n)])  # (n, *shape)
        flat    = stacked.view(n, -1)                 # (n, d_key)
        sorted_flat, _ = torch.sort(flat, dim=0)
        trimmed = sorted_flat[beta:n-beta]            # (n-2*beta, d_key)
        delta   = trimmed.mean(0).view(stacked.shape[1:])
        w_out[key] = (ori_weight[key] + eta * delta.to(ori_weight[key].dtype))
    for key in ori_weight:
        if ori_weight[key].dtype == torch.int64:
            w_out[key] = ori_weight[key].clone()
    logger.info(f\'[TRIMMEDMEAN] ep={epoch} n={n} beta={beta} kept={n-2*beta}\')
    return w_out

'''

insert_before = 'def FedAvg(handle):'
if insert_before not in src:
    print('ERROR: FedAvg marker not found'); sys.exit(1)
src = src.replace(insert_before, krum_tm_fns + insert_before, 1)

# ── 2. Inject normcap + CLIENT_FLAGS + Krum/TM dispatch ──────────────────────
cap_flags_dispatch = f'''\
        # [NORMCAP+FLAGS+CROSSFAM] Cap adversarial norms; log client ordering; dispatch Krum/TM.
        _lnc_target = {tn}
        _lnc_keys_nc = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_delta_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys_nc)
                _lnc_delta_norm = _lnc_delta_sq ** 0.5
                if _lnc_delta_norm > _lnc_target:
                    _lnc_scale = _lnc_target / _lnc_delta_norm
                    for k in _lnc_keys_nc:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_scale
                        ).to(w_locals[_lnc_i][k].dtype)
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f\'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]\')
        # [END NORMCAP+FLAGS+CROSSFAM]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: aggregation marker not found'); sys.exit(1)
src = src.replace(marker, cap_flags_dispatch + marker, 1)

# ── 3. Replace else: Aggregation(...) with dispatch including Krum/TM ─────────
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
    print('ERROR: aggregation else-block not found'); sys.exit(1)
src = src.replace(old_else, new_else, 1)

with open(alg_path, 'w') as f:
    f.write(src)
print(f'patch applied: krum/trimmedmean + normcap + flags (defense={defense}, target={tn})', flush=True)
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
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
# Krum params
p['krum_f']                       = 1
p['krum_m']                       = 1
# TrimmedMean params
p['trim_beta']                    = 1
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok defense=$DEFENSE', flush=True)
PYEOF

LOG=$OUT_DIR/step5_xfam2_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Post-training: extract ASR trajectory + Krum/TM stats ──────────────────
python3 -I - <<PYEOF2
import re, sys
log = open('$LOG').read()
lines_out = [f'=== step5_xfam2: defense=$DEFENSE cap=$TARGET_NORM seed=1 ===', '']

# ASR trajectory
asr_pat = re.compile(r'epoch (\d+).*?clean_target_rate=([\d.]+).*?attack_success_rate=([\d.]+)', re.S)
ctrl_asr = 1.9 / 100  # no-poison baseline ~2%
asr_rows = []
for m in re.finditer(r'Epoch (\d+).*?clean_target_rate=([\d.]+).*?attack_success_rate=([\d.]+)', log):
    ep, ctr, asr = int(m.group(1)), float(m.group(2)), float(m.group(3))
    net = max(0, asr - ctrl_asr)
    asr_rows.append((ep, ctr, asr, net))

# Also try lowercase
if not asr_rows:
    for m in re.finditer(r'epoch[= ](\d+).*?clean_target_rate[= ]([\d.]+).*?attack_success_rate[= ]([\d.]+)', log, re.I):
        ep, ctr, asr = int(m.group(1)), float(m.group(2)), float(m.group(3))
        net = max(0, asr - ctrl_asr)
        asr_rows.append((ep, ctr, asr, net))

if asr_rows:
    lines_out.append('  ASR trajectory (net = raw - 1.9% baseline):')
    for ep, ctr, asr, net in asr_rows[::50] + [asr_rows[-1]]:
        lines_out.append(f'    ep={ep:4d}  clean_acc={ctr:.3f}  asr={asr:.3f}  net_asr={net:.3f}')
    final = asr_rows[-1]
    lines_out.append(f'  FINAL ep={final[0]}: net_asr={final[3]:.3f}')
else:
    lines_out.append('  [no ASR rows found in log]')

# Krum scores (if krum)
krum_adv, krum_ben = [], []
for m in re.finditer(r'\[KRUM\] ep=(\d+) scores=\[([^\]]+)\] selected=\[([^\]]+)\]', log):
    scores_str = m.group(2).split(',')
    scores = [float(s.strip()) for s in scores_str]
    sel_str = m.group(3).split(',')
    sel = [int(s.strip()) for s in sel_str]
    lines_out.append(f'  [krum] ep={m.group(1)} scores={scores[:3]}... selected={sel}')

# TrimmedMean summary
tm_matches = re.findall(r'\[TRIMMEDMEAN\] ep=(\d+) n=(\d+) beta=(\d+) kept=(\d+)', log)
if tm_matches:
    lines_out.append(f'  [trimmedmean] n={tm_matches[0][1]} beta={tm_matches[0][2]} kept={tm_matches[0][3]} (per round)')

out = '\n'.join(lines_out)
print(out)
with open('$OUT_DIR/step5_xfam2_${DEF_TAG}_summary.txt', 'w') as f:
    f.write(out + '\n')
print('Saved summary.')
PYEOF2

# ── Distributional analysis ──────────────────────────────────────────────────
# Krum: use Krum scores as the detection statistic (lower = more trusted by Krum).
# TrimmedMean: use per-client norm as proxy (trimmed-mean is per-coordinate, not per-client).
# Both: use CLIENT_FLAGS + norm distribution (same as FLAME analysis).
python3 $REPO/jobs/step5_distrib_analysis.py \
    --defense flame \
    --log     $LOG \
    --out     $OUT_DIR/step5_xfam2_${DEF_TAG}_gate.txt \
    --cap     $TARGET_NORM \
    2>&1 | tee $OUT_DIR/step5_xfam2_${DEF_TAG}_gate.log

echo "Done."
echo "  Log:     $LOG"
echo "  Summary: $OUT_DIR/step5_xfam2_${DEF_TAG}_summary.txt"
echo "  Gate:    $OUT_DIR/step5_xfam2_${DEF_TAG}_gate.txt"
