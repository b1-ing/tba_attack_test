#!/usr/bin/env python3
"""Step 2c: Pooling gain analysis.

For one fixed model (warmup checkpoint), and for each trigger variant, report:
  - mean |activation change| at conv1 output (after BN+ReLU, shape 128×32×32)
  - mean |activation change| after global average pooling (shape 512×1×1)
  - ratio: pooling_change / conv1_change  (the "pooling gain")

All versus a random sample of clean non-target test images.
"""
import argparse, csv, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--n-samples',  type=int, default=500)
ap.add_argument('--amplitude',  type=float, default=0.1145,
                help='Raw-space L∞ amplitude for additive triggers (default 0.1145 ≈ t1 baseline)')
ap.add_argument('--seed',       type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)

import torch, yaml
import torch.nn.functional as F
import main as ref_main
import trigger_utils as tu
from handle import Handle
from models.ResNet8 import ResNet8
from device import device
import datetime
print(f'trigger_utils VERSION={tu.VERSION}', flush=True)

ref_main.seed(args.seed)

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['is_poison'] = False
params['poison_client_num'] = 0

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
                params=params, name=params.get('name', params['type']))
handle.load_data()
TARGET = handle.target
H, W, C = 32, 32, 3
amp = args.amplitude  # raw-space amplitude (used as trigger_amplitude_raw)

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt_path  = os.path.join(warmup_dir, 'warmup_ckpt.pt')

model = ResNet8(num_classes=10).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
model.eval()
print(f'Loaded {ckpt_path}', flush=True)

# ── Get test images ──────────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs)
all_lbls = torch.cat(all_lbls)
nt_mask = all_lbls != TARGET
nt_imgs = all_imgs[nt_mask][:args.n_samples].to(device)
print(f'Using {len(nt_imgs)} non-target images', flush=True)

# ── Activation hooks ─────────────────────────────────────────────────────────
_acts = {}

def _hook_conv1(module, inp, out):
    _acts['conv1'] = out.detach().cpu()  # (N, 128, 32, 32)

def _hook_gap(module, inp, out):
    _acts['gap'] = out.detach().cpu()    # (N, 512, 1, 1)

model.conv1  # top-level conv, NOT the block conv1
h1 = model.bn1 # register after bn1+relu; but easier: hook the linear1 input via avgpool

# Hook after the first conv+bn+relu (conv1 → bn1 → relu in ResNet8.forward)
# ResNet8.forward: out = F.relu(self.bn1(self.conv1(x)))  [line 61]
# After avgpool: out = self.avgpool(out)  [line 65]
# Register on bn1's output (which goes through relu in functional call — can't hook relu directly)
# Instead hook conv1's output pre-relu, then after avgpool.
# Use a forward hook on avgpool and a manual relu on conv1 output.

h_gap = model.avgpool.register_forward_hook(_hook_gap)

def get_conv1_act(imgs):
    """Run only through conv1→bn1→relu and return activations."""
    with torch.no_grad():
        return F.relu(model.bn1(model.conv1(imgs))).cpu()

def get_all_acts(imgs):
    with torch.no_grad():
        _ = model(imgs)
    conv1_act = get_conv1_act(imgs)
    gap_act   = _acts['gap'].squeeze(-1).squeeze(-1)  # (N, 512)
    return conv1_act, gap_act

# ── Trigger definitions via trigger_utils (raw [0,1] space) ──────────────────
patch_mask = tu.get_patch_mask(params.get('0_poison_pattern', []), device)

def _sin_amp_raw(T):
    """Sinusoidal amplitude in raw space matching L2_raw of checker at amp."""
    ss = np.sum(np.sin(2*np.pi/T * np.arange(H))**2)
    return float(amp * np.sqrt(H / ss))  # float32-safe: Python float × np.float32 stays float32

def make_raw_trigger(name):
    """Return (C,H,W) float32 numpy delta in raw [0,1] space."""
    hw_arr = ((np.arange(H).reshape(H,1) + np.arange(W).reshape(1,W)) % 2).astype(np.float32)*2-1
    if name == 'checker':
        return tu.make_raw_delta('checker', amplitude=amp)
    elif name == 'stripes':
        rows_s = (np.arange(H) % 2).astype(np.float32)*2.0-1.0
        pat = np.stack([np.broadcast_to(rows_s.reshape(H,1),(H,W))*amp]*C)
        return pat.astype(np.float32)
    elif name == 'corner4x4':
        pat = np.zeros((C,H,W),np.float32)
        for r in range(4):
            for c in range(4):
                pat[:,r,c] = (((r+c)%2)*2.0-1.0)*amp
        return pat
    elif name == 'sin16':
        a = _sin_amp_raw(16)
        row_v = np.sin(2*np.pi/16*np.arange(H)).astype(np.float32)*a
        return np.stack([np.broadcast_to(row_v.reshape(H,1),(H,W))]*C)
    elif name == 'sin32':
        a = _sin_amp_raw(32)
        row_v = np.sin(2*np.pi/32*np.arange(H)).astype(np.float32)*a
        return np.stack([np.broadcast_to(row_v.reshape(H,1),(H,W))]*C)
    elif name == 'noise':
        return tu.make_raw_delta('noise', amplitude=amp, seed=7)
    elif name == 'checker_patch_l2':
        return tu.make_raw_delta('checker', amplitude=_em_amp)
    elif name == 'eps4':
        return tu.make_raw_delta('eps4')
    elif name == 'eps8':
        return tu.make_raw_delta('eps8')
    elif name == 'patch':
        return None
    else:
        raise ValueError(name)

def apply_trigger(name, imgs):
    """Apply trigger in raw [0,1] space via trigger_utils."""
    if name == 'patch':
        return tu.apply_patch_norm(imgs, patch_mask)
    else:
        delta = make_raw_trigger(name)
        return tu.apply_additive_norm(imgs, delta)

# ── Main computation ─────────────────────────────────────────────────────────
STD_RAW = torch.tensor(tu.STD).view(1, 3, 1, 1)

# ── Pre-compute patch L2_raw for energy-matched checker amplitude ─────────────
_patch_trig_pre = tu.apply_patch_norm(nt_imgs, patch_mask)
_patch_delta_raw = (_patch_trig_pre - nt_imgs).cpu() * STD_RAW
_patch_l2_raw = _patch_delta_raw.flatten(1).norm(dim=1).mean().item()
_em_amp = tu.checker_amplitude_for_l2(_patch_l2_raw)
print(f'Patch L2_raw={_patch_l2_raw:.5f}  energy-matched checker amp={_em_amp:.5f}', flush=True)

conv1_clean, gap_clean = get_all_acts(nt_imgs)
conv1_clean_mean = conv1_clean.abs().mean().item()
gap_clean_mean   = gap_clean.abs().mean().item()

TRIGGERS = ['checker', 'stripes', 'corner4x4', 'sin16', 'sin32', 'noise',
            'checker_patch_l2', 'eps4', 'eps8', 'patch']

os.makedirs(args.out, exist_ok=True)
csv_path = os.path.join(args.out, 'step5_pooling_gain.csv')

fields   = ['trigger', 'conv1_clean_mean', 'conv1_trig_mean', 'conv1_delta_mean',
            'gap_clean_mean',  'gap_trig_mean',  'gap_delta_mean',
            'gap_over_l2', 'l2_raw', 'linf_raw']

print(f'\n{"trigger":>18}  {"conv1_Δ":>10}  {"gap_Δ":>10}  {"gap/L2":>8}  {"L2_raw":>8}  {"Linf_raw":>10}')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for name in TRIGGERS:
        trig_imgs = apply_trigger(name, nt_imgs)

        conv1_trig, gap_trig = get_all_acts(trig_imgs)

        c1_delta = (conv1_trig - conv1_clean).abs().mean().item()
        gp_delta = (gap_trig - gap_clean).abs().mean().item()

        # L2 and L∞ in raw [0,1] space via trigger_utils unnormalization
        delta_norm = (trig_imgs - nt_imgs).cpu()
        delta_raw  = delta_norm * STD_RAW
        l2_t    = delta_raw.flatten(1).norm(dim=1).mean().item()
        linf_t  = delta_raw.abs().max().item()

        gap_l2  = gp_delta / (l2_t + 1e-9)

        print(f'{name:>18}  {c1_delta:>10.5f}  {gp_delta:>10.5f}  {gap_l2:>8.4f}  {l2_t:>8.4f}  {linf_t:>10.6f}')
        writer.writerow(dict(trigger=name,
                             conv1_clean_mean=round(conv1_clean_mean,5),
                             conv1_trig_mean=round(conv1_trig.abs().mean().item(),5),
                             conv1_delta_mean=round(c1_delta,5),
                             gap_clean_mean=round(gap_clean_mean,5),
                             gap_trig_mean=round(gap_trig.abs().mean().item(),5),
                             gap_delta_mean=round(gp_delta,5),
                             gap_over_l2=round(gap_l2,4),
                             l2_raw=round(l2_t,5),
                             linf_raw=round(linf_t,6)))

h_gap.remove()
print(f'\nDone. CSV: {csv_path}', flush=True)
