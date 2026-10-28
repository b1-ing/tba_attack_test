#!/bin/bash
#SBATCH --job-name=step5_mkrum
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A.out
#SBATCH --exclude=node871,node877

# Multi-Krum rerun. Standard Krum (m=1) is uninterpretable: 44% clean acc under non-IID.
# Multi-Krum (m=5 of 10) averages 5 most-central clients; clean accuracy approaches FedAvg.
#
# CORRECTED PREDICTION:
#   Attacker norm = 0.057, benign norm ~1.2.
#   dist(attacker, benign_j)^2 ≈ ||benign_j||^2 ≈ 1.44 → sum_k = 7*1.44 ≈ 10
#   dist(benign_i, benign_j)^2 ≈ varies (0.5-3.0 under alpha=0.5 non-IID)
#   Observed: attacker score ep=1 was 4037 vs benign 100-200.
#   WHY so high? With m=1 the model oscillated; benign clients clustered around trained features,
#   and the attacker (near-zero, no feature learning) was far from everyone.
#   Prediction: Multi-Krum (m=5) still selects 5 benign clients, NEVER the attacker.
#   Expected: clean acc ~70-75%, net ASR ~0% (attacker never contributes).
#
# CRITERION IMPLICATION:
#   The attack criterion is NOT "be small" — it's "sit INSIDE the honest distribution on the
#   defense's statistic." For Krum (distance-to-neighbours), small norm = DISTANT = OUTLIER.
#   The attack fails Krum's criterion. Krum is the boundary/negative case.
#
# Config: n=10, f=1, k=7, m=5, cap=0.057, checker, alpha=0.5, seed=1, 600 rounds.

TARGET_NORM=0.057
KRUM_M=5

echo "step5_mkrum job $SLURM_JOB_ID on $(hostname)"
echo "Multi-Krum m=$KRUM_M cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_mkrum_${SLURM_JOB_ID}
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

# ── Patch: Krum impl + normcap + CLIENT_FLAGS ─────────────────────────────────
export SCRATCH_DIR TARGET_NORM KRUM_M
python3 -I - <<'PATCHEOF'
import os, sys
scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
km       = int(os.environ['KRUM_M'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

krum_fn = '''
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
    logger.info(f\'[KRUM] ep={epoch} m={m} scores={[round(s,2) for s in scores]} selected={selected}\')
    w_out = {key: ori_weight[key].clone() for key in ori_weight}
    for key in keys:
        delta = sum((w_locals[i][key] - ori_weight[key]).float() for i in selected) / m
        w_out[key] = (ori_weight[key] + eta * delta.to(ori_weight[key].dtype))
    for key in ori_weight:
        if ori_weight[key].dtype == torch.int64:
            w_out[key] = ori_weight[key].clone()
    return w_out

'''

insert_before = 'def FedAvg(handle):'
if insert_before not in src:
    print('ERROR: FedAvg marker not found'); sys.exit(1)
src = src.replace(insert_before, krum_fn + insert_before, 1)

cap_flags = f'''\
        # [NORMCAP+FLAGS+MKRUM] Cap adversarial norms + CLIENT_FLAGS.
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
        # [END NORMCAP+FLAGS+MKRUM]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: aggregation marker not found'); sys.exit(1)
src = src.replace(marker, cap_flags + marker, 1)

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
print(f'multi-krum patch applied (m={km}, target={tn})', flush=True)
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
p['krum_f']                       = 1
p['krum_m']                       = $KRUM_M
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print(f'yaml ok krum_m=$KRUM_M', flush=True)
PYEOF

LOG=$OUT_DIR/step5_mkrum_m${KRUM_M}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense krum 2>&1 | tee $LOG

echo "Training done."

python3 -I - <<PYEOF2
import re, numpy as np
log = open('$LOG').read()
lines_out = [f'=== step5_mkrum: Multi-Krum m=$KRUM_M cap=$TARGET_NORM seed=1 ===', '']
ctrl_asr = 1.9 / 100

clean_rows, asr_rows = [], []
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', log)
# Pair them up: False then True per epoch
i = 0
ep = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        ep += 1
        ca  = float(evals[i][1]) / 100
        asr = float(evals[i+1][1]) / 100
        net = max(0, asr - ctrl_asr)
        clean_rows.append(ca)
        asr_rows.append((ep, ca, asr, net))
    i += 2

if asr_rows:
    lines_out.append('  Trajectory (every 50 evals):')
    for row in asr_rows[::50] + [asr_rows[-1]]:
        lines_out.append(f'    eval={row[0]:3d}  clean={row[1]:.3f}  asr={row[2]:.3f}  net_asr={row[3]:.3f}')
    final = asr_rows[-1]
    lines_out.append(f'  FINAL: clean_acc={final[1]:.3f}  net_asr={final[3]:.3f}')
    lines_out.append(f'  Mean clean acc: {np.mean(clean_rows):.3f}')
else:
    lines_out.append('  [no eval rows found]')

# Krum selection of adversary
sel_adv = 0
sel_rounds = 0
for m in re.finditer(r'\[KRUM\] ep=(\d+) m=\d+ scores=\[([^\]]+)\] selected=\[([^\]]+)\]', log):
    ep_k = int(m.group(1))
    sel  = list(map(int, m.group(3).split(',')))
    sel_rounds += 1
    # client flags for this round
adv_flag_pat = re.compile(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]')
adv_rounds_with_adv = set()
for mf in adv_flag_pat.finditer(log):
    ep_f = int(mf.group(1))
    mask = list(map(int, mf.group(3).split(',')))
    if 1 in mask:
        adv_rounds_with_adv.add(ep_f)
lines_out.append(f'  Rounds with adversary present: {len(adv_rounds_with_adv)}')
lines_out.append(f'  Total aggregation rounds logged: {sel_rounds}')

out = '\n'.join(lines_out)
print(out)
with open('$OUT_DIR/step5_mkrum_m${KRUM_M}_summary.txt', 'w') as f:
    f.write(out + '\n')
print('Saved.')
PYEOF2

echo "Done. Log: $LOG"
