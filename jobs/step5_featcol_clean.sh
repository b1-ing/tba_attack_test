#!/bin/bash
#SBATCH --job-name=step5_fcc
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Feature-collision on a CLEAN (non-backdoored) model — fixes circularity.
#
# Step 1: Train ResNet8 on CIFAR-10 (standard supervised, no FL, no poisoning) for 25 epochs.
# Step 2: For checker, stripes, noise — compute phi(x+delta)-phi(x) feature direction and
#         its cosine with each of the 10 class FC weight rows.
#         Also compute triggered-image logit push and predicted-class histogram.
#
# Prediction (causal, pre-backdoor):
#   checker -> low collision with ALL classes (uncontested feature space) -> implantable
#   stripes -> high collision with auto/truck (horizontal gratings = vehicle features) -> contested
#   noise   -> high collision with some class -> contested
#
# If the clean model predicts which triggers work, the mechanism is causal and measurable
# without any backdoor training.

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
[[ -x "$PYTHON" ]] || { echo "FATAL: anaconda3 python missing"; exit 1; }
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

REPO=$HOME/tba_attack_test
OUT_DIR=$HOME/ffl_step5
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

echo "step5_fcc job $SLURM_JOB_ID on $(hostname)"

$PYTHON - <<PYEOF
import sys, warnings, os
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, '$REPO/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

DATA_DIR   = '$DATA_DIR'
CKPT_PATH  = '$OUT_DIR/step5_clean_resnet8.pt'
OUT_PATH   = '$OUT_DIR/step5_featcol_clean_results.txt'
AMPLITUDE  = 0.1145
N_IMAGES   = 1000
EPOCHS     = 25
BATCH_SIZE = 128
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# CIFAR-10 normalisation
mean_t = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
std_t  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
mean_np = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
std_np  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)

# ── Load CIFAR-10 ─────────────────────────────────────────────────────────────
import pickle, os

def load_cifar10_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1,3,32,32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'], dtype=np.int64)
    return imgs, labels

train_imgs, train_labels = [], []
for i in range(1, 6):
    im, lb = load_cifar10_batch(os.path.join(DATA_DIR, f'data_batch_{i}'))
    train_imgs.append(im); train_labels.append(lb)
train_imgs   = np.concatenate(train_imgs)
train_labels = np.concatenate(train_labels)

test_imgs, test_labels = load_cifar10_batch(os.path.join(DATA_DIR, 'test_batch'))

def normalise(imgs_np):
    return (imgs_np - mean_np) / std_np

train_X = torch.tensor(normalise(train_imgs))
train_y = torch.tensor(train_labels)
test_X  = torch.tensor(normalise(test_imgs))
test_y  = torch.tensor(test_labels)

train_loader = DataLoader(TensorDataset(train_X, train_y),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ── Train clean model ─────────────────────────────────────────────────────────
model = ResNet8(num_classes=10).to(device)
opt   = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit  = nn.CrossEntropyLoss()

print(f"Training clean ResNet8 for {EPOCHS} epochs...")
for ep in range(1, EPOCHS+1):
    model.train()
    total, correct = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    sched.step()
    if ep % 5 == 0 or ep == EPOCHS:
        model.eval()
        with torch.no_grad():
            test_out = model(test_X.to(device))
            test_acc = (test_out.argmax(1) == test_y.to(device)).float().mean().item()
        print(f"  ep={ep:3d}  train_acc={correct/total:.3f}  test_acc={test_acc:.3f}")

torch.save(model.state_dict(), CKPT_PATH)
print(f"Clean model saved: {CKPT_PATH}")

# ── Feature collision analysis ────────────────────────────────────────────────
model.eval()

# Hook penultimate features (input to linear1)
_penult = {}
model.linear1.register_forward_pre_hook(
    lambda m, inp: _penult.update({'feat': inp[0].detach()})
)

W_fc = model.linear1.weight.detach().cpu().numpy()          # (10, 512)
W_fc_norm = W_fc / (np.linalg.norm(W_fc, axis=1, keepdims=True) + 1e-9)
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

imgs_norm = normalise(test_imgs[:N_IMAGES])

BATCH = 128
def get_features_logits(imgs_np):
    feats, logits = [], []
    for i in range(0, len(imgs_np), BATCH):
        xb = torch.tensor(imgs_np[i:i+BATCH]).to(device)
        with torch.no_grad():
            out = model(xb)
        feats.append(_penult['feat'].cpu().numpy().copy())
        logits.append(out.cpu().numpy().copy())
    return np.concatenate(feats), np.concatenate(logits)

print("\nComputing clean features...")
feats_clean, logits_clean = get_features_logits(imgs_norm)
clean_acc = (logits_clean.argmax(1) == test_labels[:N_IMAGES]).mean()
print(f"Clean model test accuracy (first {N_IMAGES}): {clean_acc:.3f}")

lines = []
lines.append("=== Feature Collision Analysis — CLEAN (non-backdoored) model ===")
lines.append(f"Clean ResNet8 trained {EPOCHS} epochs supervised (no FL, no poison)")
lines.append(f"N={N_IMAGES} test images, amplitude={AMPLITUDE}")
lines.append(f"Clean accuracy: {clean_acc:.3f}")
lines.append(f"FC shape: {W_fc.shape}")
lines.append("")

results = {}
for trig in ['checker', 'stripes', 'noise']:
    print(f"Processing trigger: {trig}")
    delta     = make_raw_delta(trig, AMPLITUDE)          # (3,32,32) raw [0,1]
    delta_norm = delta / std_np                          # normalised
    imgs_trig  = np.clip(imgs_norm + delta_norm, -4.0, 4.0)

    feats_trig, logits_trig = get_features_logits(imgs_trig)

    feat_diff      = feats_trig - feats_clean            # (N, 512)
    feat_diff_mean = feat_diff.mean(axis=0)              # (512,)
    feat_dir       = feat_diff_mean / (np.linalg.norm(feat_diff_mean) + 1e-9)
    per_img_norm   = np.linalg.norm(feat_diff, axis=1)

    collisions     = W_fc_norm @ feat_dir                # (10,)
    logit_diff     = (logits_trig - logits_clean).mean(axis=0)
    pred_trig      = logits_trig.argmax(1)

    results[trig] = {
        'feat_norm': float(per_img_norm.mean()),
        'collisions': collisions,
        'logit_diff': logit_diff,
        'pred_trig': pred_trig,
        'max_abs_col': float(np.max(np.abs(collisions))),
        'max_abs_cls': CLASSES[int(np.argmax(np.abs(collisions)))],
        'target_col': float(collisions[0]),   # airplane = target class (c=0)
    }

    lines.append(f"=== {trig} trigger ===")
    lines.append(f"  mean ||phi(x+d)-phi(x)||: {per_img_norm.mean():.4f}")
    lines.append(f"  class-feature collision cos(trigger_dir, W_fc[c]):")
    for c, cls in enumerate(CLASSES):
        lines.append(f"    c={c} {cls:12s}: {collisions[c]:+.4f}")
    lines.append(f"  max |collision|: {np.max(np.abs(collisions)):.4f}  "
                 f"class={CLASSES[int(np.argmax(np.abs(collisions)))]}")
    lines.append(f"  target (airplane) collision: {collisions[0]:+.4f}")
    lines.append(f"  logit shift (mean over images):")
    for c, cls in enumerate(CLASSES):
        lines.append(f"    c={c} {cls:12s}: {logit_diff[c]:+.4f}")
    lines.append(f"  triggered pred dist: {np.bincount(pred_trig, minlength=10).tolist()}")
    lines.append("")

# ── Summary tables ────────────────────────────────────────────────────────────
lines.append("=== Collision Summary (CLEAN model) ===")
hdr = "  {:10s}  ".format("trigger") + "  ".join(f"{c[:5]:>6s}" for c in CLASSES)
lines.append(hdr)
for trig in ['checker', 'stripes', 'noise']:
    row = "  {:10s}  ".format(trig) + "  ".join(f"{v:+.3f}" for v in results[trig]['collisions'])
    lines.append(row)

lines.append("")
lines.append("=== Logit Push Summary (CLEAN model) ===")
lines.append(hdr)
for trig in ['checker', 'stripes', 'noise']:
    row = "  {:10s}  ".format(trig) + "  ".join(f"{v:+.3f}" for v in results[trig]['logit_diff'])
    lines.append(row)

lines.append("")
lines.append("=== Contest score: max |collision with any class| (lower = more uncontested) ===")
for trig in ['checker', 'stripes', 'noise']:
    r = results[trig]
    lines.append(f"  {trig:10s}: max_col={r['max_abs_col']:.4f} ({r['max_abs_cls']}), "
                 f"target_col={r['target_col']:+.4f}")

out = '\n'.join(lines)
print(out)
with open(OUT_PATH, 'w') as f:
    f.write(out + '\n')
print(f"\nSaved: {OUT_PATH}")
PYEOF

echo "Done."
