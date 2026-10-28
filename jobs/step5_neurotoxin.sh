#!/bin/bash
#SBATCH --job-name=step5_ntx
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Neurotoxin head-to-head vs no-selection norm-capped attack.
#
# Neurotoxin: zero out the top-NTX_TOP_FRAC of coordinates by |mean benign delta|
#             (white-box: adversary observes current round's benign submissions),
#             then rescale masked delta to exactly TARGET_NORM.
# No-selection baseline: already run (step5_floorcap_sweep, step5_crossfam2).
#
# Arms:
#   1: FilterFL,  cap=0.057  (primary comparison — do both work at 2% benign?)
#   2: FilterFL,  cap=0.023  (floor test — does Neurotoxin implant where no-sel gives 97%?)
#   3: FilterFL,  cap=0.011  (floor test — does Neurotoxin implant where no-sel gives 52%?)
#   4: FLAME,     cap=0.057  (detectability: is sparser update more/less visible?)
#   5: FLTrust,   cap=0.057  (detectability: does sparse update hurt cosine trust?)
#
# NTX_TOP_FRAC=0.8: zero out top 80% coords (keep bottom 20%).
# Config: checker, alpha=0.5, seed=1, 600 rounds.
#
# Comparison table for no-selection (already run):
#   cap=0.057 FilterFL: 100% ASR, det=0/617
#   cap=0.023 FilterFL:  97% ASR, det=0/617
#   cap=0.011 FilterFL:  52% ASR, det=0/617
#   cap=0.057 FLAME:    100% ASR (attacker below median 99.7% rounds)
#   cap=0.057 FLTrust:  100% ASR (trust~0.997 early rounds)

CAPS=(dummy 0.057 0.023 0.011 0.057 0.057)
DEFENSES=(dummy ours ours ours flame fltrust)
DEF_TAGS=(dummy filterfl filterfl filterfl flame fltrust)
NTX_TOP_FRAC=0.8

TARGET_NORM=${CAPS[$SLURM_ARRAY_TASK_ID]}
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}
CAP_TAG=$(echo "$TARGET_NORM" | tr '.' 'p')

echo "step5_ntx job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE  cap=$TARGET_NORM  ntx_top_frac=$NTX_TOP_FRAC"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_ntx_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: Neurotoxin coordinate selection + norm cap ────────────────────────
# Replaces normcap patch: masks top-NTX_TOP_FRAC coords by |mean benign delta|,
# then rescales masked delta to exactly TARGET_NORM.
# White-box assumption: adversary observes current round's benign client deltas.
export SCRATCH_DIR TARGET_NORM NTX_TOP_FRAC
python3 -I - <<'PATCHEOF'
import os, sys

scratch      = os.environ['SCRATCH_DIR']
tn           = float(os.environ['TARGET_NORM'])
top_frac     = float(os.environ['NTX_TOP_FRAC'])
alg_path     = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

ntx_block = f'''\
        # [NEUROTOXIN] Bottom-k coordinate selection + norm cap.
        # White-box: uses current round's benign client deltas to estimate
        # which coordinates benign clients update most, then avoids those.
        _ntx_target   = {tn}
        _ntx_top_frac = {top_frac}
        _ntx_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]

        # Step 1: mean benign delta across this round's benign clients
        _ntx_ben_vecs = []
        for _ni, _nname in enumerate(agent_name_keys):
            if _nname not in adversarial_name_keys:
                _ntx_ben_vecs.append(
                    torch.cat([(w_locals[_ni][k] - ori_weight[k]).float().cpu().ravel()
                               for k in _ntx_keys]))
        if _ntx_ben_vecs:
            _ntx_mean_ben = torch.stack(_ntx_ben_vecs).mean(0)  # (D,)
        else:
            _ntx_mean_ben = None

        # Step 2: build coordinate mask (zero top-NTX_TOP_FRAC by |mean benign delta|)
        if _ntx_mean_ben is not None:
            _ntx_D      = _ntx_mean_ben.shape[0]
            _ntx_k_zero = int(_ntx_top_frac * _ntx_D)
            _, _ntx_top_idx = _ntx_mean_ben.abs().topk(_ntx_k_zero, sorted=False)
            _ntx_mask = torch.ones(_ntx_D, dtype=torch.float32)
            _ntx_mask[_ntx_top_idx] = 0.0

            for _ni, _nname in enumerate(agent_name_keys):
                if _nname in adversarial_name_keys:
                    # Flatten delta
                    _ntx_parts, _ntx_shapes, _ntx_cum = [], [], [0]
                    for k in _ntx_keys:
                        _d = (w_locals[_ni][k] - ori_weight[k]).float().cpu()
                        _ntx_parts.append(_d.ravel()); _ntx_shapes.append(_d.shape)
                        _ntx_cum.append(_ntx_cum[-1] + _d.numel())
                    _ntx_flat = torch.cat(_ntx_parts)
                    # Apply mask, rescale to target norm
                    _ntx_masked = _ntx_flat * _ntx_mask
                    _ntx_mnorm  = _ntx_masked.norm().item()
                    if _ntx_mnorm > 1e-8:
                        _ntx_masked = _ntx_masked * (_ntx_target / _ntx_mnorm)
                    # Write back per-layer
                    for _ki, k in enumerate(_ntx_keys):
                        _seg = _ntx_masked[_ntx_cum[_ki]:_ntx_cum[_ki+1]].reshape(_ntx_shapes[_ki])
                        w_locals[_ni][k] = (ori_weight[k] +
                            _seg.to(device=ori_weight[k].device, dtype=ori_weight[k].dtype))
                    logger.info(f'[NEUROTOXIN] ep={{epoch}} {{_nname}} '
                                f'pre_mask_norm={{_ntx_flat.norm().item():.4f}} '
                                f'post_mask_norm={{_ntx_mnorm:.4f}}->'
                                f'{{_ntx_target:.4f}} k_zero={{_ntx_k_zero}}')
        # [END NEUROTOXIN]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found', flush=True); sys.exit(1)
src = src.replace(marker, ntx_block + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Neurotoxin patch applied: top_frac={top_frac} target_norm={tn}', flush=True)
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
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: defense=$DEFENSE checker 600r cap=$TARGET_NORM ntx_top_frac=$NTX_TOP_FRAC', flush=True)
PYEOF

LOG=$OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}.log
CKPT=$OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}_final.pt

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

LAST_CKPT=$(ls -t saved_models/*.pt 2>/dev/null | head -1)
[[ -n $LAST_CKPT ]] && cp "$LAST_CKPT" $CKPT && echo "Checkpoint: $CKPT"

echo ""
echo "Training done."

# ── Net ASR eval ──────────────────────────────────────────────────────────────
python3 -I - <<EVALEOF
import pickle, numpy as np, torch, sys
sys.path.insert(0, '$SCRATCH_DIR/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET = 0; AMP_RAW = 0.1145

try:
    sd = torch.load('$CKPT', map_location='cpu', weights_only=False)
    m = ResNet8(num_classes=10); m.load_state_dict(sd); m.eval()
except Exception as e:
    print(f'Checkpoint load failed: {e}'); sys.exit(0)

with open('$DATA_DIR/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
imgs_raw = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels   = np.array(d[b'labels'])
nt_idx   = np.where(labels != TARGET)[0]
imgs_nt  = imgs_raw[nt_idx]

def norm(x): return (x - MEAN[:,None,None]) / STD[:,None,None]
def to_t(x): return torch.tensor(np.stack([norm(i) for i in x]), dtype=torch.float32)
delta = make_raw_delta('checker', AMP_RAW, seed=1)

def predict(imgs):
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), 200):
            preds.extend(m(to_t(imgs[i:i+200])).argmax(1).tolist())
    return np.array(preds)

clean_rate = (predict(imgs_nt) == TARGET).mean()
raw_asr    = (predict(np.array([np.clip(i+delta,0,1) for i in imgs_nt])) == TARGET).mean()
net_asr    = raw_asr - clean_rate

result = (f'ntx defense=$DEFENSE cap=$TARGET_NORM top_frac=$NTX_TOP_FRAC seed=1 checker\n'
          f'clean_target_rate: {clean_rate:.4f}\n'
          f'raw_asr:           {raw_asr:.4f}\n'
          f'net_asr:           {net_asr:.4f}\n'
          f'n_nontarget:       {len(imgs_nt)}\n')
print(result)
with open('$OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}_netasr.txt', 'w') as f:
    f.write(result)
EVALEOF

# ── Summary ───────────────────────────────────────────────────────────────────
python3 -I - <<SUMEOF
import re

log_path = '$LOG'
defense  = '$DEFENSE'
cap      = $TARGET_NORM
top_frac = $NTX_TOP_FRAC

with open(log_path) as f:
    lines = f.readlines()

asr_lines    = [l for l in lines if 'Global_Test_Poison:True'  in l]
clean_lines  = [l for l in lines if 'Global_Test_Poison:False' in l]
def_lines    = [l for l in lines if 'count_all_attack' in l]
ntx_lines    = [l for l in lines if '[NEUROTOXIN]' in l]
flame_lines  = [l for l in lines if 'FLAME ep=' in l]
trust_lines  = [l for l in lines if 'FLTrust ep=' in l]

asr_final   = asr_lines[-1].strip()   if asr_lines   else 'N/A'
clean_final = clean_lines[-1].strip() if clean_lines  else 'N/A'

tot_atk = tot_def = cor_def = 0
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)', def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))
det_rate = cor_def / tot_atk if tot_atk > 0 else float('nan')
fp_rate  = (tot_def - cor_def) / tot_atk if tot_atk > 0 else float('nan')

# Extract pre/post mask norms from NTX log lines
pre_norms, post_norms = [], []
for l in ntx_lines:
    pm = re.search(r'pre_mask_norm=(\S+)', l)
    qm = re.search(r'post_mask_norm=(\S+)->(\S+)', l)
    if pm: pre_norms.append(float(pm.group(1)))
    if qm: post_norms.append(float(qm.group(1)))

ntx_summary = 'N/A'
if pre_norms:
    ntx_summary = (f'{len(pre_norms)} adv-rounds  '
                   f'pre_mask_norm mean={sum(pre_norms)/len(pre_norms):.4f}  '
                   f'post_mask_norm(before_rescale) mean={sum(post_norms)/len(post_norms):.4f}  '
                   f'target={cap:.4f}')

# FLAME median norms
flame_summary = 'N/A'
if flame_lines:
    medians = [float(re.search(r'median_norm=(\S+)', l).group(1))
               for l in flame_lines if re.search(r'median_norm=(\S+)', l)]
    if medians:
        flame_summary = (f'{len(medians)} rounds  '
                         f'median mean={sum(medians)/len(medians):.3f}  '
                         f'min={min(medians):.3f}  '
                         f'attacker_cap={cap:.3f}  '
                         f'always_below={all(cap < mn for mn in medians)}')

# FLTrust trust sums
trust_summary = 'N/A'
if trust_lines:
    sums = [float(re.search(r'sum=(\S+)', l).group(1))
            for l in trust_lines if re.search(r'sum=(\S+)', l)]
    if sums:
        trust_summary = f'{len(sums)} rounds  trust_sum mean={sum(sums)/len(sums):.3f}'

# ASR trajectory
asr_traj = [(i+1, l.strip()) for i, l in enumerate(asr_lines) if (i+1) % 100 == 0]

out = f'''=== Neurotoxin: defense={defense}  cap={cap}  top_frac={top_frac}  seed=1  checker ===
Final ASR (raw):   {asr_final}
Final clean:       {clean_final}
Adv det rate:      {cor_def}/{tot_atk} = {det_rate:.4f}  (FilterFL)
FP blocked:        {tot_def-cor_def}/{tot_atk} = {fp_rate:.4f}
NTX norm info:     {ntx_summary}
FLAME clip info:   {flame_summary}
FLTrust info:      {trust_summary}
ASR every 100r:    {' | '.join(f'r{r*6}:{l[-8:]}' for r,l in asr_traj)}

No-selection baseline (same cap={cap}, same defense={defense}):
  cap=0.057 FilterFL: 100% ASR, det=0/617
  cap=0.023 FilterFL:  97% ASR, det=0/617
  cap=0.011 FilterFL:  52% ASR, det=0/617
  cap=0.057 FLAME:    100% ASR, always_below_median=False (0.3% rounds above)
  cap=0.057 FLTrust:  100% ASR
'''
print(out)
with open('$OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}_summary.txt', 'w') as f:
    f.write(out)
SUMEOF

echo "Done."
echo "  Log:     $LOG"
echo "  Net ASR: $OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}_netasr.txt"
echo "  Summary: $OUT_DIR/step5_ntx_${DEF_TAG}_${CAP_TAG}_summary.txt"
