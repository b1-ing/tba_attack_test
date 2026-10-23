#!/usr/bin/env python3
"""W_fc drift analysis for a single trigger arm.

For each round with a saved adv_local_ep{N}.pt:
  1. Margin decomposition: fix phi@r50 / fix W@r50
  2. Manifold distance: cosine(phi_probe, mean_phi_natural_class0)
  3. Trigger phi distance: cosine(mean_phi_triggered, mean_phi_natural_class0)

All evaluated in train mode (no .eval()), matching FilterFL Stage 3.
"""
import argparse, os, sys, pickle
import numpy as np
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--trigger',   required=True)
ap.add_argument('--probe-dir', required=True)
ap.add_argument('--ffl-dir',   required=True)
ap.add_argument('--data-dir',  required=True)
ap.add_argument('--out',       required=True)
ap.add_argument('--rounds',    nargs='+', type=int, default=[50, 150, 330, 400])
ap.add_argument('--k',         type=int, default=20)
args = ap.parse_args()

sys.path.insert(0, args.ffl_dir)
from models.ResNet8 import ResNet8

TARGET = 0
device = torch.device('cpu')

# ── model helpers ─────────────────────────────────────────────────────────────
def load_model(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    return m   # train mode

def penultimate(m, x):
    with torch.no_grad():
        out = m.conv1(x); out = m.bn1(out); out = torch.relu(out)
        out = m.layer1(out); out = m.layer2(out); out = m.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        return out.view(out.size(0), -1)   # (N, D)

def get_Wfc(m):
    return m.linear1.weight.detach()   # (10, D)

def margin_from(W, phi):
    logits = W @ phi
    tl = logits[TARGET].item()
    other = torch.cat([logits[:TARGET], logits[TARGET+1:]])
    rl = other.max().item()
    return tl, rl, tl - rl

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
def load_cifar_test(data_dir, n_target=64, n_nontarget=32):
    path = os.path.join(data_dir, 'test_batch')
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'])
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:,None,None]
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:,None,None]
    def norm(x): return torch.from_numpy((x - mean) / std)
    ti = np.where(labels == TARGET)[0][:n_target]
    ni = np.where(labels != TARGET)[0][:n_nontarget]
    return norm(imgs[ti]), norm(imgs[ni])

clean_target, clean_nontarget = load_cifar_test(args.data_dir)

# ── trigger application ───────────────────────────────────────────────────────
import trigger_utils as tu

def apply_trigger(imgs_norm, trigger):
    trig = trigger.lower().replace('-', '_')
    if trig in ('checker', 'checker_energy_matched'):
        delta = tu.make_raw_delta('checker', amplitude=0.1145)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'stripes':
        delta = tu.make_raw_delta('stripes', amplitude=0.1145)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'noise':
        delta = tu.make_raw_delta('noise', amplitude=0.1145)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'eps4':
        delta = tu.make_raw_delta('checker', amplitude=4/255)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'eps8':
        delta = tu.make_raw_delta('checker', amplitude=8/255)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'eps16':
        delta = tu.make_raw_delta('checker', amplitude=16/255)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'eps32':
        delta = tu.make_raw_delta('checker', amplitude=32/255)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig == 'eps64':
        delta = tu.make_raw_delta('checker', amplitude=64/255)
        return tu.apply_additive_norm(imgs_norm, delta)
    elif trig in ('patch', 'corner4x4', 'corner4x4_energy_matched'):
        positions = [[0,0],[0,1],[0,2],[0,3],[0,6],[0,7],[0,8],[0,9],
                     [3,0],[3,1],[3,2],[3,3],[3,6],[3,7],[3,8],[3,9]]
        mask = tu.get_patch_mask(positions, device='cpu')
        return tu.apply_patch_norm(imgs_norm, mask, raw_value=1.0)
    else:
        raise ValueError(f'Unknown trigger: {trigger}')

triggered = apply_trigger(clean_nontarget, args.trigger)

# ── trigger property: spatial extent (fraction of pixels modified) ────────────
def spatial_extent(trigger_name):
    """Fraction of pixels nonzero in the trigger delta / mask."""
    t = trigger_name.lower().replace('-', '_')
    if 'patch' in t or 'corner' in t:
        positions = [[0,0],[0,1],[0,2],[0,3],[0,6],[0,7],[0,8],[0,9],
                     [3,0],[3,1],[3,2],[3,3],[3,6],[3,7],[3,8],[3,9]]
        return len(positions) / (32 * 32)
    else:
        return 1.0  # global (all pixels)

# ── trigger property: frequency (checker-band fraction) ──────────────────────
def checker_freq_frac(trigger_name):
    """Fraction of spectral energy at Nyquist (checker) frequency."""
    delta_raw = apply_trigger(
        torch.zeros(1, 3, 32, 32), trigger_name
    ) - torch.zeros(1, 3, 32, 32)
    arr = delta_raw[0].mean(0).numpy()   # (32,32) mean over channels
    fft = np.abs(np.fft.fft2(arr)) ** 2
    total = fft.sum()
    chk = fft[16, 16] if total > 0 else 0.0
    return float(chk / (total + 1e-12))

# ── load probe ────────────────────────────────────────────────────────────────
probe_path = os.path.join(args.probe_dir, 'pixel_probe_ep0050.pt')
if not os.path.exists(probe_path):
    print(f'ERROR: no frozen probe at {probe_path}', flush=True)
    sys.exit(1)
frozen_probe = torch.load(probe_path, map_location='cpu', weights_only=False)
if frozen_probe.dim() == 3:
    frozen_probe = frozen_probe.unsqueeze(0)

# ── per-round analysis ────────────────────────────────────────────────────────
results = []
phi_50, W_50 = None, None

for rnd in args.rounds:
    path = os.path.join(args.probe_dir, f'adv_local_ep{rnd:04d}.pt')
    if not os.path.exists(path):
        results.append({'rnd': rnd, 'missing': True})
        continue

    m = load_model(path)
    phi_r = penultimate(m, frozen_probe).mean(0)   # (D,)
    W_r   = get_Wfc(m)                              # (10, D)

    if rnd == args.rounds[0] or phi_50 is None:
        phi_50 = phi_r.clone()
        W_50   = W_r.clone()

    _, _, actual = margin_from(W_r,  phi_r)
    _, _, mW     = margin_from(W_r,  phi_50)   # fix phi, vary W
    _, _, mphi   = margin_from(W_50, phi_r)    # fix W, vary phi

    # manifold distance: cosine(phi_probe, mean_phi_natural_target)
    phi_nat = penultimate(m, clean_target).mean(0)
    manifold_dist = F.cosine_similarity(phi_r.unsqueeze(0),
                                        phi_nat.unsqueeze(0)).item()

    # trigger phi distance (triggered images)
    phi_trig = penultimate(m, triggered).mean(0)
    trig_dist = F.cosine_similarity(phi_trig.unsqueeze(0),
                                    phi_nat.unsqueeze(0)).item()

    # fa: cosine(phi_probe, W_fc[target])
    fa = F.cosine_similarity(phi_r.unsqueeze(0),
                              W_r[TARGET].unsqueeze(0)).item()

    results.append({
        'rnd': rnd, 'actual': actual, 'mW': mW, 'mphi': mphi,
        'fa': fa, 'manifold_dist': manifold_dist, 'trig_dist': trig_dist,
    })
    print(f'  rnd={rnd}: actual={actual:.3f}  mW={mW:.3f}  mphi={mphi:.3f}  '
          f'fa={fa:.3f}  manifold={manifold_dist:.3f}  trig={trig_dist:.3f}',
          flush=True)

# Trigger-level properties
sp_ext  = spatial_extent(args.trigger)
chk_frac = checker_freq_frac(args.trigger)

# Drift direction: sign and size of (actual_r400 - actual_r50)
r0 = next((r for r in results if not r.get('missing') and r['rnd'] == args.rounds[0]), None)
r_last = next((r for r in reversed(results) if not r.get('missing')), None)
drift = (r_last['actual'] - r0['actual']) if (r0 and r_last) else float('nan')
drift_W   = (r_last['mW']   - r0['mW'])   if (r0 and r_last) else float('nan')
drift_phi = (r_last['mphi'] - r0['mphi']) if (r0 and r_last) else float('nan')

# Summary line
summary = {
    'trigger': args.trigger,
    'spatial_extent': sp_ext,
    'checker_freq_frac': chk_frac,
    'drift_actual': drift,
    'drift_W': drift_W,
    'drift_phi': drift_phi,
    'manifold_dist_r50': r0.get('manifold_dist', float('nan')) if r0 else float('nan'),
    'trig_dist_r50': r0.get('trig_dist', float('nan')) if r0 else float('nan'),
}

# Write output
lines = [f'trigger: {args.trigger}']
lines.append(f'spatial_extent: {sp_ext:.4f}')
lines.append(f'checker_freq_frac: {chk_frac:.4f}')
lines.append('')
lines.append(f'  {"Rnd":>5}  {"actual":>8}  {"mW":>8}  {"mphi":>8}  '
             f'{"fa":>7}  {"manifold":>9}  {"trig_sim":>9}')
lines.append('  ' + '-'*68)
for r in results:
    if r.get('missing'):
        lines.append(f'  {r["rnd"]:>5}  (no file)')
    else:
        lines.append(f'  {r["rnd"]:>5}  {r["actual"]:>8.3f}  {r["mW"]:>8.3f}  '
                     f'{r["mphi"]:>8.3f}  {r["fa"]:>7.4f}  '
                     f'{r["manifold_dist"]:>9.4f}  {r["trig_dist"]:>9.4f}')
lines.append('')
lines.append(f'drift (actual r50->last): {drift:+.3f}  '
             f'W-driven: {drift_W:+.3f}  phi-driven: {drift_phi:+.3f}')
lines.append(f'manifold_dist at r50: {summary["manifold_dist_r50"]:.4f}')
lines.append(f'trig_dist at r50:     {summary["trig_dist_r50"]:.4f}')

out_text = '\n'.join(lines)
print('\n' + out_text, flush=True)
with open(args.out, 'w') as f:
    f.write(out_text + '\n')
print(f'\nSaved to {args.out}', flush=True)
