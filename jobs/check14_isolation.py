#!/usr/bin/env python3
"""Check 14: Isolating the patch-vs-BLN confound.

The patch trigger (local pixel replacement) is detected by Stage-3 (s3≈0.225),
but band-limited additive triggers at the same frequency are not (s3≈0).
Three confounds distinguish them: modification type (replacement vs additive),
spatial extent (16 px vs full image), and amplitude (≈0.7 vs 0.5).

This experiment uses a 2×2 factorial + amplitude control to isolate each factor:

  Trigger             | Type        | Extent   | Amplitude | Prediction if...
  --------------------|-------------|----------|-----------|------------------
  patch               | replacement | local    | ~0.7/px   | [KNOWN: s3≈0.225]
  additive_local      | additive    | local    | 0.5/px    | type matters → s3≈0
  additive_local_hi   | additive    | local    | 1.0/px    | amplitude controls → s3≈patch
  replace_global      | replacement | global   | 0.7/px    | extent matters → s3≈0
  bln_0.45            | additive    | global   | 0.5/px    | [KNOWN: s3≈0]

All five triggers are at the same approximate spectral frequency (median_f≈0.45),
matching the patch's spectral profile.

Decisions:
  - additive_local s3≈0, additive_local_hi s3≈0.225  → amplitude is the key variable
  - additive_local s3≈0.225 regardless of amplitude  → locality explains detectability
  - replace_global s3≈0                               → replacement mechanism alone insufficient; locality required
  - replace_global s3≈0.225                           → replacement mechanism sufficient; locality irrelevant
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--clients',    type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',      type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--n-test',     type=int, default=500)
ap.add_argument('--alpha',      type=float, default=0.5)
ap.add_argument('--global-seed',type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import yaml

import main as ref_main
import train as train_mod
from gan import gan_defense
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['seed'] = args.global_seed
params['dirichlet_alpha'] = args.alpha
params['defence_method'] = 'ours'
ref_main.seed(params['seed'])

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET = handle.target
print(f'Target class: {TARGET}', flush=True)

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'Warmup checkpoint not found: {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
global_state = copy.deepcopy(handle.model.state_dict())
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

test_imgs_list, test_lbls_list = [], []
for imgs, lbls in handle.test_data:
    test_imgs_list.append(imgs)
    test_lbls_list.append(lbls)
    if sum(x.size(0) for x in test_imgs_list) >= args.n_test:
        break
test_imgs = torch.cat(test_imgs_list)[:args.n_test].to(device)
print(f'Test batch: {test_imgs.shape}', flush=True)

C, H, W = 3, 32, 32

# ── Build trigger patterns ────────────────────────────────────────────────────

# Patch mask: same 16-pixel positions as the real patch trigger
_cpu = torch.device('cpu')
_patch_mask_t = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)  # (C,H,W) bool
_patch_mask_np = _patch_mask_t.numpy().astype(bool)   # True at the 16 trigger pixels
print(f'Patch mask: {_patch_mask_np[0].sum()} pixels active per channel, '
      f'{_patch_mask_np.sum()} total', flush=True)


def _additive_local_pattern(amplitude):
    """Add `amplitude` to patch pixel positions, 0 elsewhere. (C, H, W)."""
    pat = np.zeros((C, H, W), dtype=np.float32)
    pat[_patch_mask_np] = amplitude
    return pat


def _replace_global_pattern():
    """Continuous sinusoidal pattern at K_norm≈0.44 applied to ALL pixels.

    Uses integer wavenumber k_x=10 so the pattern has a single spectral spike
    at K_norm = 10 / sqrt(16^2+16^2) ≈ 0.442, matching the patch's median_f≈0.445.
    Values are in [0, 1] (replacement, not additive).

    Binary thresholding was avoided because it adds odd harmonics that shift
    the spectral median to lower frequencies (~0.225 vs target 0.445).
    """
    k_x = 10  # integer wavenumber → K_norm = 10/k_max ≈ 0.442
    col_idx = np.arange(W, dtype=np.float32)
    # Zero-mean cosine — no DC offset so the spectral median lands at k_x=10.
    # (0.5 + 0.5*cos has mean=0.5; its DC term dominates and shifts median to ≈0.)
    stripe = 0.5 * np.cos(2 * np.pi * k_x * col_idx / W)  # (W,) ∈ [-0.5, 0.5]
    pattern = np.stack([np.tile(stripe, (H, 1))] * C, axis=0)  # (C, H, W)
    return pattern.astype(np.float32)  # used as: image = pattern (replacement)


_additive_local_05  = _additive_local_pattern(0.5)
_additive_local_10  = _additive_local_pattern(1.0)
_replace_global_pat = _replace_global_pattern()

print(f'replace_global pattern: active pixels per channel = '
      f'{(_replace_global_pat[0] > 0).sum()}/{H*W}', flush=True)


def _verify_spectral_median(pat_np):
    """Quick median normalised frequency of pat_np (C,H,W)."""
    k_max = np.sqrt((H/2)**2 + (W/2)**2)
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    KX, KY = np.meshgrid(kx, ky)
    K_norm = np.sqrt(KX**2 + KY**2) / k_max
    power = np.zeros((H, W))
    for ch in range(C):
        f = np.fft.fft2(pat_np[ch])
        power += np.abs(np.fft.fftshift(f))**2
    power /= C
    n_bins = 100
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_p = np.zeros(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (K_norm >= lo) & (K_norm < hi)
        bin_p[i] = power[m].sum() if m.any() else 0.0
    cum = np.cumsum(bin_p); total = cum[-1] + 1e-30
    idx = np.searchsorted(cum / total, 0.5)
    return float(centers[min(idx, n_bins - 1)])


f_patch   = _verify_spectral_median(_patch_mask_np.astype(np.float32))
f_al05    = _verify_spectral_median(_additive_local_05)
f_al10    = _verify_spectral_median(_additive_local_10)
f_rg      = _verify_spectral_median(_replace_global_pat)
print(f'Spectral median_f — patch mask: {f_patch:.3f}, '
      f'additive_local_05: {f_al05:.3f}, additive_local_10: {f_al10:.3f}, '
      f'replace_global: {f_rg:.3f}', flush=True)


# ── Measurement function ───────────────────────────────────────────────────────

def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def measure(client_id, trigger_name, seed):
    """Run one poisoned-training measurement for the named trigger."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate'] = 0.50

    import train as tm
    _orig = tm.apply_pattern_batch_gpu

    if trigger_name == 'patch':
        # Use the original patch mechanism (no monkey-patch needed)
        p['trigger_type'] = 'patch'
        def patched(images, pms, index, handle_, poison_mask):
            return _orig(images, pms, index, handle_, poison_mask)

    elif trigger_name == 'additive_local_05':
        p['trigger_type'] = '_custom'
        pat_np = _additive_local_05
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            poisoned = images + pat_t.unsqueeze(0)
            poisoned_labels = torch.full(
                (images.size(0),), TARGET, device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), poisoned_labels

    elif trigger_name == 'additive_local_hi':
        p['trigger_type'] = '_custom'
        pat_np = _additive_local_10
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            poisoned = torch.clamp(images + pat_t.unsqueeze(0), -1.0, 1.0)
            poisoned_labels = torch.full(
                (images.size(0),), TARGET, device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), poisoned_labels

    elif trigger_name == 'replace_global':
        p['trigger_type'] = '_custom'
        pat_np = _replace_global_pat
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            # Replace: overwrite ALL pixel positions with pattern values
            replaced = pat_t.unsqueeze(0).expand_as(images)
            poisoned_labels = torch.full(
                (images.size(0),), TARGET, device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), replaced, images), poisoned_labels

    elif trigger_name == 'bln_0.45':
        # Use check13's BLN mechanism at f=0.45 as a reference
        p['trigger_type'] = '_custom'
        from jobs_util import make_bln  # fallback: inline below
        raise NotImplementedError

    else:
        raise ValueError(f'Unknown trigger: {trigger_name}')

    tm.apply_pattern_batch_gpu = patched

    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    try:
        w_local = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id,
            params=p,
            model=model,
            dataset=handle.train_data[client_id],
            handle=handle,
            poison=True,
        )
    finally:
        tm.apply_pattern_batch_gpu = _orig

    W_l_target = w_local['linear1.weight'][TARGET]

    torch.manual_seed(42); np.random.seed(42)
    g_imgs = gan_defense(global_state, w_local, params, the_epoch=1)
    probe_imgs = g_imgs[0]

    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(w_local)
    m.eval()
    with torch.no_grad():
        probs_probe = torch.softmax(m(probe_imgs), dim=1)
        s3 = probs_probe.mean(0)[TARGET].item()
        feats_probe = features(m, probe_imgs)

    feat_align = torch.nn.functional.cosine_similarity(
        feats_probe.mean(0, keepdim=True), W_l_target.unsqueeze(0)).item()

    # Apply trigger to test images for eval
    with torch.no_grad():
        if trigger_name == 'patch':
            mask_t = _patch_mask_t.to(device).bool()
            mask_b = mask_t.unsqueeze(0).expand_as(test_imgs)
            triggered = torch.where(mask_b, torch.tensor(1.0, device=device, dtype=test_imgs.dtype), test_imgs)
        elif trigger_name in ('additive_local_05', 'additive_local_hi'):
            amp = 0.5 if trigger_name == 'additive_local_05' else 1.0
            pat_t = torch.from_numpy(_additive_local_05 if amp == 0.5 else _additive_local_10).to(device)
            triggered = test_imgs + pat_t.unsqueeze(0)
            if trigger_name == 'additive_local_hi':
                triggered = torch.clamp(triggered, -1.0, 1.0)
        elif trigger_name == 'replace_global':
            pat_t = torch.from_numpy(_replace_global_pat).to(device)
            triggered = pat_t.unsqueeze(0).expand_as(test_imgs).clone()

        feats_trig = features(m, triggered)
        probs_trig = torch.softmax(m(triggered), dim=1)
        asr_proxy  = probs_trig.mean(0)[TARGET].item()

    trigger_feat_al = torch.nn.functional.cosine_similarity(
        feats_trig.mean(0, keepdim=True), W_l_target.unsqueeze(0)).item()

    return {
        'trigger': trigger_name,
        'client':  client_id,
        'seed':    seed,
        'feat_align':     feat_align,
        'trigger_feat_al':trigger_feat_al,
        'evasion_gap':    trigger_feat_al - feat_align,
        's3':             s3,
        'asr_proxy':      asr_proxy,
    }


TRIGGERS = ['patch', 'additive_local_05', 'additive_local_hi', 'replace_global']
COLS = ['trigger', 'client', 'seed',
        'feat_align', 'trigger_feat_al', 'evasion_gap', 's3', 'asr_proxy']

rows = []
print(f'\n{"trigger":>20}  {"client":>6}  {"seed":>4}  '
      f'{"feat_al":>8}  {"trig_al":>8}  {"gap":>7}  {"s3":>7}  {"asr_px":>8}', flush=True)
print('-' * 85)

for trigger_name in TRIGGERS:
    for client_id in args.clients:
        for seed in args.seeds:
            r = measure(client_id, trigger_name, seed)
            rows.append(r)
            print(f"{r['trigger']:>20}  {r['client']:>6}  {r['seed']:>4}  "
                  f"{r['feat_align']:>8.4f}  {r['trigger_feat_al']:>8.4f}  "
                  f"{r['evasion_gap']:>7.4f}  {r['s3']:>7.4f}  {r['asr_proxy']:>8.4f}",
                  flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst) / len(lst) if lst else float('nan')

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>20}  {'feat_al':>8}  {'trig_al':>8}  {'gap':>7}  {'s3':>7}  {'asr_px':>8}")
print('-'*70)
for t in TRIGGERS:
    tr = [r for r in rows if r['trigger'] == t]
    print(f"{t:>20}  {mn([r['feat_align'] for r in tr]):>8.4f}  "
          f"{mn([r['trigger_feat_al'] for r in tr]):>8.4f}  "
          f"{mn([r['evasion_gap'] for r in tr]):>7.4f}  "
          f"{mn([r['s3'] for r in tr]):>7.4f}  "
          f"{mn([r['asr_proxy'] for r in tr]):>8.4f}")

print('\n=== Interpretation matrix ===')
patch_s3 = mn([r['s3'] for r in rows if r['trigger'] == 'patch'])
al05_s3  = mn([r['s3'] for r in rows if r['trigger'] == 'additive_local_05'])
alhi_s3  = mn([r['s3'] for r in rows if r['trigger'] == 'additive_local_hi'])
rg_s3    = mn([r['s3'] for r in rows if r['trigger'] == 'replace_global'])

print(f"  patch s3={patch_s3:.4f}")
print(f"  additive_local_05 s3={al05_s3:.4f}  additive_local_hi s3={alhi_s3:.4f}")
print(f"  replace_global s3={rg_s3:.4f}")

thresh = patch_s3 * 0.3  # "similar to BLN" threshold
if al05_s3 < thresh and alhi_s3 < thresh:
    print('  → REPLACEMENT TYPE is necessary (additive local evades regardless of amplitude)')
elif al05_s3 < thresh and alhi_s3 >= thresh * 3:
    print('  → AMPLITUDE explains difference (additive local at patch amplitude ≈ detectable)')
else:
    print('  → LOCALITY drives detection (additive local ≈ detectable; type does not matter)')

if rg_s3 < thresh:
    print('  → LOCALITY is necessary (global replacement evades despite patch mechanism)')
else:
    print('  → REPLACEMENT MECHANISM alone is sufficient (global replacement detectable)')

csv_path = os.path.join(args.out, 'check14_isolation.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    wr.writerows(rows)
print(f'\nSaved to {csv_path}')
