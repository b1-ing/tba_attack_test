#!/bin/bash
#SBATCH --job-name=step5_mkrms
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Multi-Krum seeds 2-3. Verifies the steady-state "attacker selected at convergence" result
# from seed 1 (23799771) in a healthy model (clean acc >70%).
#
# Arms:
#   1: seed=2
#   2: seed=3
#
# Config: n=10, f=1, k=7, m=5, cap=0.057, checker, alpha=0.5, 600 rounds.

TARGET_NORM=0.057
KRUM_M=5
SEEDS=(dummy 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "step5_mkrms job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Multi-Krum m=$KRUM_M seed=$SEED cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_mkrms_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: Krum + normcap + CLIENT_FLAGS ──────────────────────────────────────
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
    print('ERROR: FedAvg not found'); sys.exit(1)
src = src.replace(insert_before, krum_fn + insert_before, 1)

cap_flags = f'''\
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
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
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
print(f'patch applied (m={km}, target={tn})', flush=True)
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
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['krum_f']                       = 1
p['krum_m']                       = $KRUM_M
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print(f'yaml ok seed=$SEED krum_m=$KRUM_M', flush=True)
PYEOF

LOG=$OUT_DIR/step5_mkrum_m${KRUM_M}_s${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense krum 2>&1 | tee $LOG

echo "Training done."

# ── Analysis ──────────────────────────────────────────────────────────────────
python3 -I - <<PYEOF2
import re, os
import numpy as np

log      = open('$LOG').read()
out_file = '$OUT_DIR/step5_mkrum_m${KRUM_M}_s${SEED}_summary.txt'
CTRL     = 0.019

flags = {}
for m in re.finditer(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]', log):
    ep   = int(m.group(1))
    mask = list(map(int, m.group(3).split(',')))
    flags[ep] = [i for i,v in enumerate(mask) if v==1]

krum = {}
for m in re.finditer(r'\[KRUM\] ep=(\d+) m=(\d+) scores=\[([^\]]+)\] selected=\[([^\]]+)\]', log):
    ep     = int(m.group(1))
    scores = list(map(float, m.group(3).split(',')))
    sel    = list(map(int, m.group(4).split(',')))
    krum[ep] = (scores, sel)

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', log)
pairs = []
i = 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

lines = [f'=== Multi-Krum m=$KRUM_M seed=$SEED cap=$TARGET_NORM ===', '']

if pairs:
    final_c, final_a = pairs[-1]
    net = max(0, final_a - CTRL)
    lines.append(f'  Final: clean={final_c:.3f}  asr={final_a:.3f}  net_asr={net:.3f}')
    lines.append(f'  Mean clean acc (all evals): {np.mean([c for c,a in pairs]):.3f}')

# Per-round attacker score at rounds where clean acc > 70%
lines.append('')
lines.append('  Rounds with clean acc >70%: attacker Krum score vs benign range')
high_ca_rounds = [(ep, pairs[ep-1][0]) for ep in sorted(krum.keys())
                  if ep-1 < len(pairs) and pairs[ep-1][0] > 0.70
                  and ep in flags and flags[ep]]
lines.append(f'  n_rounds_clean>70%_with_adv: {len(high_ca_rounds)}')
sel_count = 0
for ep, ca in high_ca_rounds[:20]:
    scores, sel = krum[ep]
    for ai in flags[ep]:
        if ai >= len(scores): continue
        adv_s = scores[ai]
        ben   = [s for i,s in enumerate(scores) if i not in flags[ep]]
        rank  = sorted(scores).index(adv_s) + 1
        in_s  = ai in sel
        if in_s: sel_count += 1
        lines.append(f'    ep={ep:4d} clean={ca:.2f} adv={adv_s:.2f} '
                     f'ben=[{min(ben):.2f},{max(ben):.2f}] rank={rank}/10 '
                     f'{"SEL" if in_s else "out"}')
if high_ca_rounds:
    tot = len(high_ca_rounds)
    lines.append(f'  Selected in {sel_count}/{tot} clean>70% rounds ({100*sel_count/tot:.0f}%)')

out = '\n'.join(lines)
print(out)
with open(out_file, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {out_file}')
PYEOF2

echo "Done. Log: $LOG"
