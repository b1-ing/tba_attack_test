#!/usr/bin/env python3
"""Check 13: Frequency–evasion trade-off curve.

Sweeps a family of band-limited additive triggers from median normalised
frequency 0.25 to 0.99, measuring Stage-3 detection (s3, feat_align) and
backdoor learnability (asr_proxy, trigger_feat_al) for each.

Frequency normalisation matches check11_spectral.py:
  K_norm in [0, 1], where 1 = Nyquist corner (sqrt((H/2)^2+(W/2)^2)).
  axis-aligned Nyquist ≈ 0.707, checker ≈ 0.99.

Design:
  For each target frequency f ∈ F_TARGETS, a band-limited trigger is built by
  sampling white noise → 2D FFT → Gaussian bandpass at K_norm=f → IFFT →
  L∞-normalise to ±amplitude.  Bandwidth FWHM = 0.10, narrow enough that
  adjacent bands don't bleed across the GAN probe cutoff (~0.5).

Reference triggers (patch, checker, noise) are included for comparison.

Metrics per (client, trigger, seed):
  median_f        actual spectral median of the built trigger
  feat_align      cos(probe_features, W_fc[target]) — Stage-3 input
  trigger_feat_al cos(triggered_features, W_fc[target])
  evasion_gap     trigger_feat_al − feat_align
  s3              p(target class) on GAN probe images (Stage-3 score)
  asr_proxy       p(target class) on triggered test images (learnability)
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--clients',    type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',      type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--amplitude',  type=float, default=0.50)
ap.add_argument('--bandwidth',  type=float, default=0.10,
                help='Gaussian bandpass FWHM in normalised frequency units')
ap.add_argument('--n-test',     type=int,   default=500)
ap.add_argument('--alpha',      type=float, default=0.5)
ap.add_argument('--global-seed',type=int,   default=1)
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

# Test batch for asr_proxy / trigger_feat_al
test_imgs_list, test_lbls_list = [], []
for imgs, lbls in handle.test_data:
    test_imgs_list.append(imgs)
    test_lbls_list.append(lbls)
    if sum(x.size(0) for x in test_imgs_list) >= args.n_test:
        break
test_imgs = torch.cat(test_imgs_list)[:args.n_test].to(device)
print(f'Test batch: {test_imgs.shape}', flush=True)

# ── Image shape ───────────────────────────────────────────────────────────────
dummy_p = {'trigger_type': 'checker', 'trigger_amplitude': 0.5}
_chk = train_mod._get_additive_trigger('checker', dummy_p, (3, 32, 32), device)
C, H, W = 3, 32, 32

# Frequency grid (same normalisation as check11_spectral.py)
ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)
k_max = np.sqrt((H / 2)**2 + (W / 2)**2)
K_NORM = K / k_max          # (H, W) normalised radial frequency [0, 1]
K_NORM_FFT = np.fft.ifftshift(K_NORM)  # FFT-ordered version

# Target frequencies for the sweep
F_TARGETS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
             0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def radial_median_f(pattern_np):
    """Compute spectral median normalised frequency of a (C, H, W) pattern."""
    power_sum = np.zeros((H, W))
    for ch in range(C):
        f = np.fft.fft2(pattern_np[ch])
        power_sum += np.abs(np.fft.fftshift(f)) ** 2
    power_mean = power_sum / C
    # Radially bin
    n_bins = 100
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_power = np.zeros(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (K_NORM >= lo) & (K_NORM < hi)
        bin_power[i] = power_mean[m].sum() if m.any() else 0.0
    cum = np.cumsum(bin_power)
    total = cum[-1] + 1e-30
    med_idx = np.searchsorted(cum / total, 0.5)
    return float(bin_centers[min(med_idx, n_bins - 1)])


def make_bandlimited_trigger(f_target, seed=42):
    """Band-limited trigger with Gaussian bandpass centred at f_target.

    Bandwidth (FWHM) = args.bandwidth.  Returns (C, H, W) float32 numpy array
    normalised to L∞ = args.amplitude.
    """
    sigma = args.bandwidth / 2.355  # Gaussian sigma from FWHM
    mask = np.exp(-0.5 * ((K_NORM_FFT - f_target) / sigma) ** 2)

    rng = np.random.RandomState(seed)
    result = np.zeros((C, H, W), dtype=np.float32)
    for c in range(C):
        noise = rng.randn(H, W).astype(np.float32)
        F_noise = np.fft.fft2(noise)
        img = np.real(np.fft.ifft2(F_noise * mask)).astype(np.float32)
        result[c] = img

    max_val = np.abs(result).max()
    if max_val > 1e-9:
        result = result / max_val * args.amplitude
    return result


def apply_bandlimited(imgs, pattern_np):
    """Add band-limited pattern to a batch (N, C, H, W) tensor."""
    pattern_t = torch.from_numpy(pattern_np).to(imgs.device)
    return imgs + pattern_t.unsqueeze(0)


def apply_reference(imgs, trigger_type, p):
    """Apply patch / checker / noise trigger to a batch."""
    if trigger_type in ('checker', 'noise'):
        additive = train_mod._get_additive_trigger(trigger_type, p, (C, H, W), imgs.device)
        return imgs + additive.unsqueeze(0)
    # patch: replace pattern pixels
    mask = train_mod.get_pattern_mask(p, 0, (C, H, W), imgs.device).bool()
    mask_b = mask.unsqueeze(0).expand_as(imgs)
    val = 1.0 if trigger_type != 'inverted' else 0.0
    return torch.where(mask_b, torch.tensor(val, device=imgs.device, dtype=imgs.dtype), imgs)


def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def measure_bandlimited(client_id, f_target, seed):
    """One measurement: bandlimited trigger at f_target on client_id."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate']       = 0.50
    p['trigger_type']      = 'bandlimited'
    p['trigger_amplitude'] = args.amplitude

    # Build trigger pattern
    pattern_np = make_bandlimited_trigger(f_target, seed=seed)
    actual_f   = radial_median_f(pattern_np)

    # Inject trigger into training by monkey-patching apply_pattern_batch_gpu
    import train as tm
    _orig_apply = tm.apply_pattern_batch_gpu

    def _patched_apply(images, pms, index, handle_, poison_mask):
        pm_exp = poison_mask.view(-1, 1, 1, 1)
        pattern_t = torch.from_numpy(pattern_np).to(images.device)
        poisoned = images + pattern_t.unsqueeze(0)
        images_after = torch.where(pm_exp.expand_as(images).bool(), poisoned, images)
        poisoned_labels = torch.full(
            (images.size(0),), TARGET, device=images.device, dtype=torch.long)
        return images_after, poisoned_labels

    tm.apply_pattern_batch_gpu = _patched_apply

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
        tm.apply_pattern_batch_gpu = _orig_apply

    W_l_target = w_local['linear1.weight'][TARGET]

    # Stage-3 probe
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
        feats_probe.mean(0, keepdim=True),
        W_l_target.unsqueeze(0)).item()

    # Triggered test images
    with torch.no_grad():
        triggered = apply_bandlimited(test_imgs, pattern_np)
        feats_trig  = features(m, triggered)
        probs_trig  = torch.softmax(m(triggered), dim=1)
        asr_proxy   = probs_trig.mean(0)[TARGET].item()

    trigger_feat_al = torch.nn.functional.cosine_similarity(
        feats_trig.mean(0, keepdim=True),
        W_l_target.unsqueeze(0)).item()

    return {
        'trigger_type':   f'bln_{f_target:.2f}',
        'f_target':       f_target,
        'actual_f':       actual_f,
        'client':         client_id,
        'seed':           seed,
        'feat_align':     feat_align,
        'trigger_feat_al':trigger_feat_al,
        'evasion_gap':    trigger_feat_al - feat_align,
        's3':             s3,
        'asr_proxy':      asr_proxy,
    }


def measure_reference(client_id, trigger_type, seed):
    """Reference measurement for patch / checker / noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate']       = 0.50
    p['trigger_type']      = trigger_type
    p['trigger_amplitude'] = args.amplitude

    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
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
        feats_probe.mean(0, keepdim=True),
        W_l_target.unsqueeze(0)).item()

    with torch.no_grad():
        triggered = apply_reference(test_imgs, trigger_type, p)
        feats_trig  = features(m, triggered)
        probs_trig  = torch.softmax(m(triggered), dim=1)
        asr_proxy   = probs_trig.mean(0)[TARGET].item()

    trigger_feat_al = torch.nn.functional.cosine_similarity(
        feats_trig.mean(0, keepdim=True),
        W_l_target.unsqueeze(0)).item()

    # Compute actual spectral median for reference trigger
    if trigger_type == 'checker':
        rows_ = np.arange(H).reshape(H, 1)
        cols_ = np.arange(W).reshape(1, W)
        pat = (((rows_ + cols_) % 2).astype(np.float32) * 2.0 - 1.0)
        pat = np.stack([pat * args.amplitude] * C, axis=0)
    elif trigger_type == 'noise':
        rng = np.random.default_rng(12345)
        pat = rng.random((C, H, W)).astype(np.float32) * 2.0 - 1.0
        pat = pat / (np.abs(pat).max() + 1e-9) * args.amplitude
    else:  # patch
        cpu_dev = torch.device('cpu')
        mask = train_mod.get_pattern_mask(p, 0, (C, H, W), cpu_dev).numpy()
        pat = mask.astype(np.float32)
    actual_f = radial_median_f(pat)

    return {
        'trigger_type':   trigger_type,
        'f_target':       actual_f,
        'actual_f':       actual_f,
        'client':         client_id,
        'seed':           seed,
        'feat_align':     feat_align,
        'trigger_feat_al':trigger_feat_al,
        'evasion_gap':    trigger_feat_al - feat_align,
        's3':             s3,
        'asr_proxy':      asr_proxy,
    }


COLS = ['trigger_type', 'f_target', 'actual_f', 'client', 'seed',
        'feat_align', 'trigger_feat_al', 'evasion_gap', 's3', 'asr_proxy']
rows = []

print(f'\n{"trigger":>16}  {"f_tgt":>6}  {"f_act":>6}  {"client":>6}  {"seed":>4}  '
      f'{"feat_al":>8}  {"trig_al":>8}  {"gap":>7}  {"s3":>7}  {"asr_px":>8}', flush=True)
print('-' * 100)

# Band-limited sweep
for f in F_TARGETS:
    for client_id in args.clients:
        for seed in args.seeds:
            r = measure_bandlimited(client_id, f, seed)
            rows.append(r)
            print(f"{r['trigger_type']:>16}  {r['f_target']:>6.2f}  {r['actual_f']:>6.3f}  "
                  f"{r['client']:>6}  {r['seed']:>4}  {r['feat_align']:>8.4f}  "
                  f"{r['trigger_feat_al']:>8.4f}  {r['evasion_gap']:>7.4f}  "
                  f"{r['s3']:>7.4f}  {r['asr_proxy']:>8.4f}", flush=True)

# Reference triggers
for ttype in ['patch', 'checker', 'noise']:
    for client_id in args.clients:
        for seed in args.seeds:
            r = measure_reference(client_id, ttype, seed)
            rows.append(r)
            print(f"{r['trigger_type']:>16}  {r['f_target']:>6.3f}  {r['actual_f']:>6.3f}  "
                  f"{r['client']:>6}  {r['seed']:>4}  {r['feat_align']:>8.4f}  "
                  f"{r['trigger_feat_al']:>8.4f}  {r['evasion_gap']:>7.4f}  "
                  f"{r['s3']:>7.4f}  {r['asr_proxy']:>8.4f}", flush=True)

# ── Summary: mean per (trigger_type, f_target) ───────────────────────────────
from itertools import groupby

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>16}  {'f_actual':>8}  {'feat_al':>8}  {'trig_al':>8}  "
      f"{'gap':>7}  {'s3':>7}  {'asr_px':>8}")
print('-' * 80)

by_type_f = {}
for r in rows:
    key = (r['trigger_type'], round(r['f_target'], 3))
    by_type_f.setdefault(key, []).append(r)

def mn(rlist, k): return np.mean([r[k] for r in rlist])

for key in sorted(by_type_f, key=lambda x: x[1]):
    rlist = by_type_f[key]
    print(f"{key[0]:>16}  {mn(rlist,'actual_f'):>8.3f}  {mn(rlist,'feat_align'):>8.4f}  "
          f"{mn(rlist,'trigger_feat_al'):>8.4f}  {mn(rlist,'evasion_gap'):>7.4f}  "
          f"{mn(rlist,'s3'):>7.4f}  {mn(rlist,'asr_proxy'):>8.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out, 'check13_freq_sweep.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    wr.writerows(rows)
print(f'\nSaved to {csv_path}')
