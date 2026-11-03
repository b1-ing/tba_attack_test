"""Optimise a trigger delta to minimise max feature-collision score.

Gradient descent on delta (C,H,W) to minimise max_c |cos(feat_shift_dir, W_fc_norm[c])|,
projected to the same L2 as checker after each step.

Saves result to ~/ffl_step5/low_col_opt_delta.npy
"""
import sys, warnings, os, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

CKPT_PATH = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_fl_clean_model.pt'
DATA_DIR  = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test/data/cifar-10-batches-py'
OUT_PATH  = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/low_col_opt_delta.npy'
AMPLITUDE = 0.1145
N_IMAGES  = 500
N_STEPS   = 800
LR        = 5e-3
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

# Load model (frozen)
model = ResNet8(num_classes=10).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['global_state'])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

_feats = {}
model.linear1.register_forward_pre_hook(lambda m, inp: _feats.update({'f': inp[0]}))

W_fc      = model.linear1.weight.detach()        # (10, 512)
W_fc_norm = F.normalize(W_fc, dim=1)             # (10, 512), unit rows
CLASSES   = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ImageNet stats used by FilterFL
mean_np = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
std_np  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)
std_t   = torch.tensor(std_np).to(device)

# Load CIFAR-10 test images
with open(os.path.join(DATA_DIR, 'test_batch'), 'rb') as fh:
    d = pickle.load(fh, encoding='bytes')
imgs_raw  = d[b'data'].reshape(-1,3,32,32).astype(np.float32)[:N_IMAGES] / 255.0
imgs_norm = (imgs_raw - mean_np) / std_np
imgs_t    = torch.tensor(imgs_norm).to(device)

# Clean features (no grad needed)
with torch.no_grad():
    model(imgs_t)
feats_clean = _feats['f'].detach().clone()  # (N, 512)

# Checker baseline
d_chk    = make_raw_delta('checker', AMPLITUDE)
l2_chk   = float(np.linalg.norm(d_chk))
print(f"checker L2 (raw): {l2_chk:.4f}")

def compute_max_col(delta_np):
    """Report max |collision| for a given delta (numpy, no grad)."""
    with torch.no_grad():
        d_t    = torch.tensor(delta_np, dtype=torch.float32).to(device)
        d_norm = d_t / std_t
        imgs_trig = torch.clamp(imgs_t + d_norm.unsqueeze(0), -4.0, 4.0)
        model(imgs_trig)
        feats_trig = _feats['f'].detach()
        feat_diff  = feats_trig - feats_clean
        feat_dir   = feat_diff.mean(0)
        feat_dir   = feat_dir / (feat_dir.norm() + 1e-9)
        cols = (W_fc_norm @ feat_dir).abs()
    return float(cols.max()), float(cols.mean()), cols.cpu().numpy()

mc, _, _ = compute_max_col(d_chk)
print(f"checker max_col: {mc:.4f}  (reference)")

# ── Optimization ──────────────────────────────────────────────────────────────
# Initialize from checker
delta = torch.tensor(d_chk.copy(), dtype=torch.float32, device=device, requires_grad=True)
optimizer = torch.optim.Adam([delta], lr=LR)

best_delta = d_chk.copy()
best_max_col = mc

for step in range(N_STEPS):
    optimizer.zero_grad()

    d_norm    = delta / std_t
    imgs_trig = torch.clamp(imgs_t + d_norm.unsqueeze(0), -4.0, 4.0)
    model(imgs_trig)
    feats_trig = _feats['f']

    feat_diff = feats_trig - feats_clean
    feat_dir  = feat_diff.mean(0)
    feat_dir  = feat_dir / (feat_dir.norm() + 1e-9)

    # Minimize mean-squared collision (smooth; drives ALL class alignments down)
    cols = W_fc_norm @ feat_dir          # (10,)
    loss = (cols ** 2).mean()
    loss.backward()
    optimizer.step()

    # Project delta to checker L2 (norm constraint)
    with torch.no_grad():
        cur_l2 = delta.data.norm()
        if cur_l2 > 1e-9:
            delta.data = delta.data * (l2_chk / cur_l2)

    if step % 100 == 0 or step == N_STEPS - 1:
        mc_cur, mean_cur, _ = compute_max_col(delta.detach().cpu().numpy())
        print(f"step={step:4d}  loss={loss.item():.5f}  max_col={mc_cur:.4f}  mean_col={mean_cur:.4f}")
        if mc_cur < best_max_col:
            best_max_col = mc_cur
            best_delta   = delta.detach().cpu().numpy().copy()

print(f"\nBest max_col achieved: {best_max_col:.4f}  (checker baseline: {mc:.4f})")
mc_final, mean_final, cols_final = compute_max_col(best_delta)
print(f"Final collision vector:")
for c, col in enumerate(cols_final):
    print(f"  {CLASSES[c]:12s}: {col:+.4f}")

np.save(OUT_PATH, best_delta.astype(np.float32))
print(f"\nSaved: {OUT_PATH}")
print(f"L2 raw: {float(np.linalg.norm(best_delta)):.4f}  (checker: {l2_chk:.4f})")
