#!/bin/bash
#SBATCH --job-name=step5_floorcap
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Floor cap sweep: push norm cap to 0.05x, 0.02x, 0.01x benign to find implantation floor.
# Benign mean norm ≈ 1.13.
# Arms: 0.05x=0.057  0.02x=0.023  0.01x=0.011
#
# Also saves all client deltas at key rounds → subspace overlap analysis.
# Post-training: net ASR eval (triggered - clean target rate).
# Config: checker, alpha=0.5, seed=1, 600 rounds, FilterFL (ours).

CAPS=(dummy 0.057 0.023 0.011)
TARGET_NORM=${CAPS[$SLURM_ARRAY_TASK_ID]}
CAP_TAG=$(echo "$TARGET_NORM" | tr '.' 'p')

echo "step5_floorcap job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "TARGET_NORM=$TARGET_NORM  tag=$CAP_TAG"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_floorcap_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
DELTA_DIR=$OUT_DIR/deltas_floorcap_${CAP_TAG}
mkdir -p $OUT_DIR $DELTA_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch 1: norm-cap on adversarial client deltas ────────────────────────────
export SCRATCH_DIR TARGET_NORM DELTA_DIR
python3 -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

cap_block = f'''\
        # [NORMCAP] Cap adversarial client update norms.
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_delta_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_delta_norm = _lnc_delta_sq ** 0.5
                if _lnc_delta_norm > _lnc_target:
                    _lnc_scale = _lnc_target / _lnc_delta_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_scale
                        ).to(w_locals[_lnc_i][k].dtype)
                    logger.info(f'[NORMCAP] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_delta_norm:.3f}}->{{_lnc_target:.3f}}')
        # [END NORMCAP]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found', flush=True); sys.exit(1)
src = src.replace(marker, cap_block + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patch 1 (normcap {tn}) applied.', flush=True)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch 1 failed"; exit 1; fi

# ── Patch 2: save all client deltas at probe_save_epochs ──────────────────────
python3 -I - <<'PATCHEOF'
import os, sys

scratch   = os.environ['SCRATCH_DIR']
delta_dir = os.environ['DELTA_DIR']
alg_path  = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

save_block = f'''\
        # [DELTA SAVE] Save per-client weight deltas for subspace overlap analysis.
        _so_epset = set(handle.params.get('probe_save_epochs', []))
        if epoch in _so_epset:
            import os as _sos
            _sdir = '{delta_dir}'
            _sos.makedirs(_sdir, exist_ok=True)
            _adv_deltas, _ben_deltas = [], []
            for _ci, _cclient in enumerate(agent_name_keys):
                _dw = {{_k: (w_locals[_ci][_k] - ori_weight[_k]).cpu().float()
                       for _k in ori_weight if ori_weight[_k].dtype != torch.int64}}
                if _cclient in adversarial_name_keys:
                    _adv_deltas.append(_dw)
                else:
                    _ben_deltas.append(_dw)
            _dpath = _sos.path.join(_sdir, f'client_deltas_ep{{epoch:04d}}.pt')
            torch.save({{'adv': _adv_deltas, 'ben': _ben_deltas,
                         'epoch': epoch, 'n_adv': len(_adv_deltas)}}, _dpath)
            logger.info(f'CLIENT_DELTAS_SAVED ep={{epoch}} n_adv={{len(_adv_deltas)}} '
                        f'n_ben={{len(_ben_deltas)}}')
        # [END DELTA SAVE]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found', flush=True); sys.exit(1)
src = src.replace(marker, save_block + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print('Patch 2 (delta save) applied.', flush=True)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch 2 failed"; exit 1; fi

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
p['probe_save_epochs']            = [50, 100, 150, 200, 300, 400, 500, 600]
p['probe_save_dir']               = '$DELTA_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker 600r cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_floorcap_${CAP_TAG}.log
CKPT=$OUT_DIR/step5_floorcap_${CAP_TAG}_final.pt

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

# Copy final checkpoint
LAST_CKPT=$(ls -t saved_models/*.pt 2>/dev/null | head -1)
if [[ -n $LAST_CKPT ]]; then
    cp "$LAST_CKPT" $CKPT && echo "Checkpoint saved: $CKPT"
fi

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
TARGET = 0
AMP_RAW = 0.1145

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
def apply_trig(imgs): return np.array([np.clip(i + delta, 0, 1) for i in imgs])

def predict(imgs):
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), 200):
            preds.extend(m(to_t(imgs[i:i+200])).argmax(1).tolist())
    return np.array(preds)

clean_rate = (predict(imgs_nt) == TARGET).mean()
raw_asr    = (predict(apply_trig(imgs_nt)) == TARGET).mean()
net_asr    = raw_asr - clean_rate

result = (f'=== floorcap net ASR: cap=$TARGET_NORM seed=1 checker ===\n'
          f'clean_target_rate: {clean_rate:.4f}\n'
          f'raw_asr:           {raw_asr:.4f}\n'
          f'net_asr:           {net_asr:.4f}\n'
          f'n_nontarget:       {len(imgs_nt)}\n')
print(result)
with open('$OUT_DIR/step5_floorcap_${CAP_TAG}_netasr.txt', 'w') as f:
    f.write(result)
EVALEOF

# ── Subspace overlap ──────────────────────────────────────────────────────────
echo "Running subspace overlap analysis..."
python3 $REPO/jobs/step5_subspace_overlap.py \
    --delta-dir $DELTA_DIR \
    --log       $LOG \
    --out       $OUT_DIR/step5_floorcap_${CAP_TAG}_overlap.txt \
    2>&1 | tee $OUT_DIR/step5_floorcap_${CAP_TAG}_overlap.log

# ── Summary ───────────────────────────────────────────────────────────────────
python3 -I - <<SUMEOF
import re, sys

log_path = '$LOG'
cap      = $TARGET_NORM
with open(log_path) as f:
    lines = f.readlines()

asr_lines  = [l for l in lines if 'Global_Test_Poison:True'  in l]
def_lines  = [l for l in lines if 'count_all_attack' in l]
asr_final  = asr_lines[-1].strip() if asr_lines else 'N/A'

tot_atk = cor_def = tot_def = 0
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)', def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))
det_rate = cor_def / tot_atk if tot_atk > 0 else float('nan')
fp_rate  = (tot_def - cor_def) / tot_atk if tot_atk > 0 else float('nan')

# ASR every 100 rounds
traj = [(i+1, l.strip()) for i, l in enumerate(asr_lines) if (i+1) % 100 == 0]

summary = (f'cap={cap:.3f} ({cap/1.13:.3f}x benign) seed=1 checker 600r FilterFL\n'
           f'final ASR (raw):   {asr_final}\n'
           f'adv det rate:      {cor_def}/{tot_atk} = {det_rate:.4f}\n'
           f'FP blocked:        {tot_def-cor_def}/{tot_atk} = {fp_rate:.4f}\n'
           f'ASR every 100r:    ' + ' | '.join(f'r{r*6}: {l[-8:]}' for r,l in traj) + '\n')
print(summary)
with open('$OUT_DIR/step5_floorcap_${CAP_TAG}_summary.txt', 'w') as f:
    f.write(summary)
SUMEOF

echo "Done."
echo "  Log:     $LOG"
echo "  Net ASR: $OUT_DIR/step5_floorcap_${CAP_TAG}_netasr.txt"
echo "  Overlap: $OUT_DIR/step5_floorcap_${CAP_TAG}_overlap.txt"
