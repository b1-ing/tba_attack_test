#!/bin/bash
#SBATCH --job-name=step5_featcol
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out

# Feature-collision analysis: for checker, stripes, noise triggers, measure:
# 1. phi(x+delta) - phi(x) direction (mean over test images) at penultimate layer
# 2. Cosine with each FC class weight row (W_fc) -> class-feature collision
# 3. Baseline logit push: mean logit output for triggered clean images
#
# Uses the FedAvg+checker checkpoint (for non-checker triggers, this model's
# FC rows are normal class features; checker's own direction will appear in
# target class row, which is informative but clearly labelled).
#
# Hypothesis: checker's trigger feature is orthogonal to all W_fc rows;
# stripes' trigger feature collides with a real class direction (horizontal
# grating = edge detector = natural class feature).

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3 python missing"; exit 1; }
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

REPO=$HOME/tba_attack_test
OUT_DIR=$HOME/ffl_step5
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

echo "step5_featcol job $SLURM_JOB_ID on $(hostname)"

$PYTHON - <<PYEOF
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Add repo to path ──────────────────────────────────────────────────────────
sys.path.insert(0, '$REPO/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

DATA_DIR   = '$DATA_DIR'
CKPT_PATH  = '$OUT_DIR/step5_fedavg_checker_a0.5_ckpt.pt'
OUT_PATH   = '$OUT_DIR/step5_featcol_results.txt'
AMPLITUDE  = 0.1145
N_IMAGES   = 1000   # test images to average over
device     = 'cpu'

# ── Load model ────────────────────────────────────────────────────────────────
model = ResNet8(num_classes=10).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
# checkpoint format: {'round': int, 'global_state': state_dict, 'rows': ...}
if isinstance(ckpt, dict) and 'global_state' in ckpt:
    model.load_state_dict(ckpt['global_state'])
elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    model.load_state_dict(ckpt['state_dict'])
elif isinstance(ckpt, dict) and any(k.startswith('conv') or k.startswith('layer') or k.startswith('linear') for k in ckpt):
    model.load_state_dict(ckpt)
else:
    raise ValueError(f"Unknown checkpoint format, keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
model.eval()
print(f"Model loaded. linear1 shape: {model.linear1.weight.shape}")

# ── Hook to extract penultimate features (pre-FC) ────────────────────────────
_penult = {}
def _hook(module, inp, out):
    _penult['feat'] = out.detach()
# avgpool → flatten is in forward: out = torch.flatten(out, 1) then linear1
# Hook the linear1 input by hooking on linear1 itself
handle = model.linear1.register_forward_pre_hook(
    lambda m, inp: _penult.update({'feat': inp[0].detach()})
)

# ── Load CIFAR-10 test images ─────────────────────────────────────────────────
import pickle, os
def load_cifar10_test(data_dir):
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'])
    return imgs, labels

imgs_raw, labels = load_cifar10_test(DATA_DIR)
# Use first N_IMAGES
imgs_raw = imgs_raw[:N_IMAGES]
labels   = labels[:N_IMAGES]

# Normalize (FilterFL uses: mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])
mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)
imgs_norm = (imgs_raw - mean) / std   # shape: (N, 3, 32, 32)

# ── FC weight rows: W_fc[c] = direction model uses for class c ───────────────
W_fc = model.linear1.weight.detach().numpy()  # (10, 512)
W_fc_norm = W_fc / (np.linalg.norm(W_fc, axis=1, keepdims=True) + 1e-9)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ── Compute phi(x+delta) and phi(x) in batches ──────────────────────────────
BATCH = 128
def get_features(imgs_np):
    feats = []
    for i in range(0, len(imgs_np), BATCH):
        batch = torch.tensor(imgs_np[i:i+BATCH])
        with torch.no_grad():
            model(batch)
        feats.append(_penult['feat'].numpy().copy())
    return np.concatenate(feats, axis=0)  # (N, 512)

def get_logits(imgs_np):
    logits = []
    for i in range(0, len(imgs_np), BATCH):
        batch = torch.tensor(imgs_np[i:i+BATCH])
        with torch.no_grad():
            out = model(batch)
        logits.append(out.numpy().copy())
    return np.concatenate(logits, axis=0)  # (N, 10)

print("Computing clean features...")
feats_clean = get_features(imgs_norm)
logits_clean = get_logits(imgs_norm)

lines = []
lines.append("=== Feature Collision Analysis ===")
lines.append(f"Model: FedAvg+checker (a0.5), N={N_IMAGES} test images")
lines.append(f"Architecture: ResNet8, penultimate dim=512, FC shape={W_fc.shape}")
lines.append("")

# Baseline: clean image logit distribution
logit_pred_clean = np.argmax(logits_clean, axis=1)
lines.append(f"Clean accuracy (top-1): {np.mean(logit_pred_clean == labels):.3f}")
lines.append("")

triggers = ['checker', 'stripes', 'noise']
results = {}

for trig in triggers:
    print(f"Processing trigger: {trig}")
    delta = make_raw_delta(trig, AMPLITUDE)   # (3, 32, 32) in [0,1] space
    # Apply in normalized space: delta_norm = delta / std (no mean shift)
    delta_norm = delta / std                  # (3, 32, 32)

    imgs_triggered = np.clip(imgs_norm + delta_norm, -3.0, 3.0)

    feats_trig  = get_features(imgs_triggered)
    logits_trig = get_logits(imgs_triggered)

    # Feature direction: mean phi(x+delta) - phi(x), then normalize
    feat_diff = feats_trig - feats_clean        # (N, 512)
    feat_diff_mean = feat_diff.mean(axis=0)     # (512,)
    feat_diff_norm = feat_diff_mean / (np.linalg.norm(feat_diff_mean) + 1e-9)

    # Per-image feature diff norm (signal strength)
    per_img_norm = np.linalg.norm(feat_diff, axis=1)

    # Collision: cos(trigger_feature_dir, W_fc[c]) for each class
    collisions = W_fc_norm @ feat_diff_norm    # (10,)

    # Logit push: mean logit shift, and predicted class of triggered images
    logit_diff_mean = (logits_trig - logits_clean).mean(axis=0)  # (10,)
    pred_trig = np.argmax(logits_trig, axis=1)

    results[trig] = {
        'feat_diff_norm_mean': float(per_img_norm.mean()),
        'collisions': collisions,
        'logit_diff': logit_diff_mean,
        'pred_trig': pred_trig,
    }

    lines.append(f"=== {trig} trigger ===")
    lines.append(f"  mean ||phi(x+d)-phi(x)||: {per_img_norm.mean():.4f}  std={per_img_norm.std():.4f}")
    lines.append(f"  class-feature collision cos(trigger_dir, W_fc[c]):")
    for c, cls in enumerate(CLASSES):
        marker = " <-- TARGET" if c == 0 else ""  # class 0 = airplane, adjust if needed
        lines.append(f"    c={c} {cls:12s}: {collisions[c]:+.4f}{marker}")
    lines.append(f"  max |collision|: {np.max(np.abs(collisions)):.4f}  class={CLASSES[np.argmax(np.abs(collisions))]}")
    lines.append(f"  logit shift (mean over images):")
    for c, cls in enumerate(CLASSES):
        lines.append(f"    c={c} {cls:12s}: {logit_diff_mean[c]:+.4f}")
    lines.append(f"  triggered pred distribution: {np.bincount(pred_trig, minlength=10).tolist()}")
    lines.append("")

# Summary comparison table
lines.append("=== Collision Summary ===")
lines.append(f"  {'trigger':10s}  " + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
for trig in triggers:
    cols = "  ".join(f"{v:+.3f}" for v in results[trig]['collisions'])
    lines.append(f"  {trig:10s}  {cols}")

lines.append("")
lines.append("=== Logit Push Summary ===")
lines.append(f"  {'trigger':10s}  " + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
for trig in triggers:
    cols = "  ".join(f"{v:+.3f}" for v in results[trig]['logit_diff'])
    lines.append(f"  {trig:10s}  {cols}")

out = '\n'.join(lines)
print(out)
with open(OUT_PATH, 'w') as f:
    f.write(out + '\n')
print(f"\nSaved: {OUT_PATH}")
PYEOF

echo "Done."
