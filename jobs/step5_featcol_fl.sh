#!/bin/bash
#SBATCH --job-name=step5_fcfl
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Step 1: Train CLEAN FL model (FedAvg, 100 participants, Dirichlet α=0.5,
#         NO poisoning, 150 rounds) using FilterFL's main.py.
# Step 2: Feature-collision analysis on that FL-trained clean model for:
#   - Existing triggers: checker, stripes, noise, sin16, corner4x4, patch
#   - Intermediate blended triggers: blend_25/50/75 (checker+stripes mix)
#     and stripes_p4 (period-4 stripes)
# Goal: confirm pre-commitment ranking holds in FL feature space, and identify
#   whether a threshold exists between 0.344 (checker) and 0.429 (stripes).

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3 python missing"; exit 1; }
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$HOME/ffl_step5/fltrust_root.pkl
OUT_DIR=$HOME/ffl_step5
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_fcfl_${SLURM_JOB_ID}

echo "step5_fcfl job $SLURM_JOB_ID on $(hostname)"

mkdir -p $SCRATCH_DIR
cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch algorithm.py to save global model at round 150 ─────────────────────
export SCRATCH_DIR OUT_DIR
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
out_dir = os.environ['OUT_DIR']
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

save_patch = f'''
        # [SAVE_CLEAN] Save aggregated global model at epoch 150
        if epoch == 150:
            import torch as _torch_sc
            _ckpt_path = '{out_dir}/step5_fl_clean_model.pt'
            _global_sd = {{k: v.clone() for k, v in w_glob.items()}}
            _torch_sc.save({{'round': epoch, 'global_state': _global_sd}}, _ckpt_path)
            logger.info(f'[SAVE_CLEAN] Saved w_glob at epoch 150 to {{_ckpt_path}}')
        # [END SAVE_CLEAN]
'''

# Patch after handle.model.load_state_dict(w_glob) — this is where the new global model is set
marker = '        handle.model.load_state_dict(w_glob)'
if marker not in src:
    print('ERROR: load_state_dict marker not found'); sys.exit(1)
src = src.replace(marker, marker + '\n' + save_patch, 1)
with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print('save_clean patch applied: compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML: clean FedAvg, no poisoning ─────────────────────────────────────────
$PYTHON -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 0        # NO adversaries
p['is_poison']                    = False    # clean training
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['epochs']                       = 150
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok: clean FedAvg, 100 clients, Dirichlet alpha=0.5, 150 rounds', flush=True)
PYEOF

LOG=$OUT_DIR/step5_fl_clean.log
cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
echo "Running clean FedAvg (150 rounds, no poisoning)..."
$PYTHON main.py --defense fedavg 2>&1 | tee $LOG
echo "FL training done."

# ── Check model was saved ─────────────────────────────────────────────────────
[[ -f $OUT_DIR/step5_fl_clean_model.pt ]] || { echo "FATAL: clean model not saved"; exit 1; }
echo "Clean FL model saved: $OUT_DIR/step5_fl_clean_model.pt"

# ── Feature-collision analysis on FL-trained clean model ──────────────────────
$PYTHON - <<PYEOF
import sys, warnings, os, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch

sys.path.insert(0, '$REPO/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

DATA_DIR  = '$DATA_DIR'
CKPT_PATH = '$OUT_DIR/step5_fl_clean_model.pt'
OUT_PATH  = '$OUT_DIR/step5_featcol_fl_results.txt'
AMPLITUDE = 0.1145
N_IMAGES  = 1000
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mean_np = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
std_np  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)

# ── Load clean FL model ───────────────────────────────────────────────────────
model = ResNet8(num_classes=10).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['global_state'])
model.eval()
print(f"Clean FL model loaded from round {ckpt['round']}")

# Hook penultimate layer
_penult = {}
model.linear1.register_forward_pre_hook(
    lambda m, inp: _penult.update({'feat': inp[0].detach()})
)

W_fc      = model.linear1.weight.detach().cpu().numpy()  # (10, 512)
W_fc_norm = W_fc / (np.linalg.norm(W_fc, axis=1, keepdims=True) + 1e-9)
CLASSES   = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ── Load CIFAR-10 test images ─────────────────────────────────────────────────
def load_cifar10_test(data_dir):
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1,3,32,32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'])
    return imgs, labels

imgs_raw, labels = load_cifar10_test(DATA_DIR)
imgs_norm        = (imgs_raw[:N_IMAGES] - mean_np) / std_np

BATCH = 128
def get_feats_logits(imgs_np):
    feats, logits = [], []
    for i in range(0, len(imgs_np), BATCH):
        xb = torch.tensor(imgs_np[i:i+BATCH]).to(device)
        with torch.no_grad():
            out = model(xb)
        feats.append(_penult['feat'].cpu().numpy().copy())
        logits.append(out.cpu().numpy().copy())
    return np.concatenate(feats), np.concatenate(logits)

print("Computing clean features...")
feats_clean, logits_clean = get_feats_logits(imgs_norm)
clean_acc = (logits_clean.argmax(1) == labels[:N_IMAGES]).mean()
print(f"Clean FL model test accuracy (first {N_IMAGES}): {clean_acc:.3f}")

# ── Build trigger dict: existing + intermediate ───────────────────────────────
def make_blend(alpha, amp=AMPLITUDE):
    """alpha*checker + (1-alpha)*stripes, normalised to same L2 as checker."""
    d_chk = make_raw_delta('checker', amp)
    d_str = make_raw_delta('stripes', amp)
    d_mix = alpha * d_chk + (1 - alpha) * d_str
    # Renormalise to checker L2
    l2_chk = float(np.linalg.norm(d_chk))
    l2_mix = float(np.linalg.norm(d_mix))
    if l2_mix > 1e-9:
        d_mix = d_mix * (l2_chk / l2_mix)
    return d_mix

def make_stripes_p4(amp=AMPLITUDE):
    """Period-4 horizontal stripes: ±amplitude per 4-row block."""
    C, H, W = 3, 32, 32
    row_sign = ((np.arange(H, dtype=int) // 2) % 2 * 2 - 1).astype(np.float32)
    pat = np.broadcast_to(row_sign.reshape(H,1), (H,W)) * amp
    return np.stack([pat.copy()] * C, axis=0)

named_triggers = {
    'checker'    : make_raw_delta('checker',  AMPLITUDE),
    'blend_75'   : make_blend(0.75),          # 75% checker, 25% stripes
    'blend_50'   : make_blend(0.50),          # 50/50
    'blend_25'   : make_blend(0.25),          # 25% checker, 75% stripes
    'stripes'    : make_raw_delta('stripes',  AMPLITUDE),
    'stripes_p4' : make_stripes_p4(),         # period-4 stripes (lower freq)
    'sin16'      : make_raw_delta('sin16',    AMPLITUDE),
    'corner4x4'  : make_raw_delta('corner4x4',AMPLITUDE),
    'noise'      : make_raw_delta('noise',    AMPLITUDE),
    'patch'      : make_raw_delta('patch',    AMPLITUDE),
}

# Known implantability (from normcap runs): checker=YES, all others=NO
known_result = {
    'checker':'YES','blend_75':'?','blend_50':'?','blend_25':'?',
    'stripes':'NO','stripes_p4':'?','sin16':'NO','corner4x4':'NO',
    'noise':'NO','patch':'NO'
}

lines = []
lines.append("=== Feature Collision — FL-trained CLEAN model (FedAvg, alpha=0.5, 150 rounds) ===")
lines.append(f"N={N_IMAGES} test images, amplitude={AMPLITUDE}")
lines.append(f"Clean FL accuracy: {clean_acc:.3f}")
lines.append("")

summary = []
for trig_name, delta_raw in named_triggers.items():
    delta_norm = delta_raw / std_np
    imgs_trig  = np.clip(imgs_norm + delta_norm, -4.0, 4.0)

    feats_trig, logits_trig = get_feats_logits(imgs_trig)

    feat_diff      = feats_trig - feats_clean
    feat_dir       = feat_diff.mean(axis=0)
    feat_dir      /= (np.linalg.norm(feat_dir) + 1e-9)
    per_img_norm   = np.linalg.norm(feat_diff, axis=1)
    collisions     = W_fc_norm @ feat_dir         # (10,)
    logit_diff     = (logits_trig - logits_clean).mean(axis=0)
    pred_trig      = logits_trig.argmax(1)

    max_col     = float(np.max(np.abs(collisions)))
    max_cls     = CLASSES[int(np.argmax(np.abs(collisions)))]
    target_col  = float(collisions[0])           # airplane = c=0
    pred_counts = np.bincount(pred_trig, minlength=10).tolist()
    dominant    = CLASSES[int(np.argmax(pred_counts))]

    summary.append({
        'name': trig_name,
        'max_col': max_col,
        'max_cls': max_cls,
        'target_col': target_col,
        'feat_norm': float(per_img_norm.mean()),
        'pred_counts': pred_counts,
        'dominant': dominant,
        'logit_diff': logit_diff,
        'known': known_result[trig_name],
    })

    lines.append(f"=== {trig_name} (implant={known_result[trig_name]}) ===")
    lines.append(f"  ||feat_diff|| mean: {per_img_norm.mean():.4f}")
    lines.append(f"  max |collision|: {max_col:.4f}  class={max_cls}")
    lines.append(f"  target col (airplane): {target_col:+.4f}")
    lines.append(f"  dominant pred: {dominant} ({max(pred_counts)}/1000)")
    lines.append(f"  collision vector: " +
                 ", ".join(f"{CLASSES[c][:4]}:{collisions[c]:+.3f}" for c in range(10)))
    lines.append(f"  logit shift:      " +
                 ", ".join(f"{CLASSES[c][:4]}:{logit_diff[c]:+.3f}" for c in range(10)))
    lines.append("")

# ── Summary table ordered by max_col ─────────────────────────────────────────
summary.sort(key=lambda x: x['max_col'])
lines.append("=== Pre-commitment ranking (sorted by max |collision|, CLEAN FL model) ===")
lines.append(f"  {'trigger':12s}  {'max_col':>8s}  {'vs_class':>10s}  {'dominant_pred':>14s}  {'implant':>7s}")
lines.append(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*7}")
for r in summary:
    lines.append(f"  {r['name']:12s}  {r['max_col']:8.4f}  {r['max_cls']:>10s}  "
                 f"{r['dominant']:>14s}  {r['known']:>7s}")

out = '\n'.join(lines)
print(out)
with open(OUT_PATH, 'w') as f:
    f.write(out + '\n')
print(f"\nSaved: {OUT_PATH}")
PYEOF

echo "Done."
