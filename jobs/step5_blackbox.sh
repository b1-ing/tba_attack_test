#!/bin/bash
#SBATCH --job-name=step5_bbox
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Black-box attack: attacker projects its update orthogonal to the PUBLIC global update,
# then caps to TARGET_NORM. No private gradient knowledge needed.
#
# Motivation (gap 2):
#   Effective rank analysis showed global-update overlap = 0.31% without explicit projection.
#   This experiment enforces the projection EXPLICITLY using only publicly-visible info
#   (the previous round's aggregated update, visible to all participants).
#   Goal: confirm (a) near-zero overlap with global direction is maintained/tightened,
#         (b) ASR still reaches the same ceiling as the baseline normcap attack,
#         (c) the threat model is strictly BLACK-BOX (no private benign gradient access).
#   This beats Neurotoxin's grey-box requirement (needs current-round benign gradients).
#
# Arms:
#   1: FilterFL cap=0.057 seed=1  (compare to baseline filterfl distributional run)
#   2: no-defense (FedAvg) cap=0.057 seed=1  (control: does projection hurt ASR?)
#
# Config: alpha=0.5, checker trigger, 600 rounds.

TARGET_NORM=0.057
DEFENSES=(dummy ours None)
DEF_TAGS=(dummy filterfl nodef)

DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

echo "step5_bbox job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_bbox_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: black-box orthogonal projection + normcap + CLIENT_FLAGS ───────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys
scratch  = os.environ['SCRATCH_DIR']
tn       = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# Inject: before aggregation marker, add:
#   1. normcap (same as baseline)
#   2. Black-box projection: subtract component along previous global update direction
#   3. CLIENT_FLAGS logging
#   4. Log overlap with global direction for each adversarial round
bbox_patch = f'''\
        # [BBOX] Black-box orthogonal projection + normcap + CLIENT_FLAGS.
        _bbox_target = {tn}
        _bbox_keys   = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        # Compute previous global update direction (public: all participants see global model).
        if not hasattr(FedAvg, '_bbox_prev_global'):
            FedAvg._bbox_prev_global = {{k: v.clone() for k, v in ori_weight.items()}}
        _bbox_prev = FedAvg._bbox_prev_global
        _bbox_g_flat = torch.cat([(ori_weight[k] - _bbox_prev[k]).float().cpu().ravel()
                                   for k in _bbox_keys])
        _bbox_g_norm = _bbox_g_flat.norm().item()
        _bbox_g_hat  = (_bbox_g_flat / _bbox_g_norm) if _bbox_g_norm > 1e-10 else _bbox_g_flat
        # Apply to each adversarial client
        for _bb_i, _bb_name in enumerate(agent_name_keys):
            if _bb_name not in adversarial_name_keys:
                continue
            # Flatten adversarial delta
            _bb_delta = torch.cat([(w_locals[_bb_i][k] - ori_weight[k]).float().cpu().ravel()
                                    for k in _bbox_keys])
            # Project out global direction (black-box: only global update used)
            if _bbox_g_norm > 1e-10:
                _bb_proj   = (_bb_delta @ _bbox_g_hat).item()
                _bb_delta  = _bb_delta - _bb_proj * _bbox_g_hat
            # Cap norm
            _bb_norm = _bb_delta.norm().item()
            if _bb_norm > _bbox_target:
                _bb_delta = _bb_delta * (_bbox_target / _bb_norm)
                _bb_norm  = _bbox_target
            # Compute overlap with global direction for logging
            _bb_ov = ((_bb_delta @ _bbox_g_hat).item() ** 2 / (_bb_norm**2 + 1e-12))
            logger.info(f\'[BBOX] ep={{epoch}} adv={{_bb_name}} \' +
                        f\'norm={{_bb_norm:.5f}} global_g_norm={{_bbox_g_norm:.4f}} \' +
                        f\'overlap_with_global={{_bb_ov:.6f}}\')
            # Write back projected+capped delta
            _bb_offset = 0
            for k in _bbox_keys:
                numel = ori_weight[k].numel()
                chunk = _bb_delta[_bb_offset:_bb_offset+numel].view(ori_weight[k].shape)
                w_locals[_bb_i][k] = (ori_weight[k] + chunk.to(device=ori_weight[k].device, dtype=ori_weight[k].dtype))
                _bb_offset += numel
        # Store current global for next round
        FedAvg._bbox_prev_global = {{k: v.clone() for k, v in ori_weight.items()}}
        # CLIENT_FLAGS
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f\'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]\')
        # [END BBOX]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, bbox_patch + marker, 1)

with open(alg_path, 'w') as f:
    f.write(src)
print(f'black-box patch applied (target={tn})', flush=True)
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
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
p['log_ben_s3']                   = ('$DEFENSE' == 'ours')
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok defense=$DEFENSE', flush=True)
PYEOF

LOG=$OUT_DIR/step5_bbox_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Post-training analysis ────────────────────────────────────────────────────
python3 -I - <<PYEOF2
import re, numpy as np
log = open('$LOG').read()
lines_out = [f'=== step5_bbox: defense=$DEFENSE cap=$TARGET_NORM seed=1 ===', '']

ctrl_asr = 1.9 / 100

# ASR trajectory
asr_rows = []
for m in re.finditer(r'[Ee]poch[= ](\d+).*?clean_target_rate[= ]([\d.]+).*?attack_success_rate[= ]([\d.]+)', log):
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
    lines_out.append('  [no ASR rows found]')

# Overlap with global direction
ov_vals = [float(m.group(1)) for m in
           re.finditer(r'overlap_with_global=([\d.eE+-]+)', log)]
if ov_vals:
    ov = np.array(ov_vals)
    lines_out.append(f'')
    lines_out.append(f'  Overlap with global update direction (post-projection):')
    lines_out.append(f'    n_rounds={len(ov)}  mean={np.mean(ov):.6f}  '
                     f'p5={np.percentile(ov,5):.6f}  p95={np.percentile(ov,95):.6f}')
    lines_out.append(f'    Baseline (no projection): 0.0031 (0.31%) from effective rank analysis')
    lines_out.append(f'    → Black-box projection {("TIGHTENS" if np.mean(ov) < 0.0031 else "does not tighten")} overlap')
else:
    lines_out.append('  [no BBOX overlap lines found]')

# FilterFL detection
det_rounds = len(re.findall(r'FEAT_ALIGN_DETECT', log))
total_fa   = len(re.findall(r'FEAT_ALIGN ep=', log))
lines_out.append(f'')
if '$DEFENSE' == 'ours':
    lines_out.append(f'  FilterFL detection: {det_rounds}/{total_fa} adversarial rounds detected')

out = '\n'.join(lines_out)
print(out)
with open('$OUT_DIR/step5_bbox_${DEF_TAG}_summary.txt', 'w') as f:
    f.write(out + '\n')
print('Saved summary.')
PYEOF2

echo "Done."
echo "  Log:     $LOG"
echo "  Summary: $OUT_DIR/step5_bbox_${DEF_TAG}_summary.txt"
