#!/usr/bin/env python3
"""Check 11 (spectral): Direct frequency-domain verification of the evasion mechanism.

Claim: the GAN generator's output is concentrated at low spatial frequencies,
while the checkerboard trigger sits at Nyquist (highest possible frequency).
Their spectral supports don't overlap, so the GAN's probes structurally cannot
excite the checkerboard feature direction.

Protocol:
  1. Run the GAN once from the warmup checkpoint → probe images.
  2. Compute the trigger additive patterns (checker, noise).
  3. Build a reference pixel-patch delta (set pattern pixels from mean→1.0).
  4. For each signal: 2D DFT per channel per image, radially average power spectrum.
  5. Report cumulative power fraction in low / mid / high frequency bands,
     and the median normalised frequency carrying 50% of the power.
"""
import argparse, copy, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--n-gan',      type=int, default=200,
                help='Number of GAN probe images to average spectrum over')
ap.add_argument('--amplitude',  type=float, default=0.50)
ap.add_argument('--seed',       type=int, default=42)
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
params['seed'] = args.seed
params['dirichlet_alpha'] = 0.5
params['defence_method'] = 'ours'
ref_main.seed(params['seed'])

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'No warmup checkpoint at {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
global_state = copy.deepcopy(handle.model.state_dict())
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

# ── 1. Generate GAN probe images ─────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)
print('Running GAN...', flush=True)
g_imgs = gan_defense(global_state, global_state, params, the_epoch=1)
probe_imgs = g_imgs[0].cpu().numpy()  # (N, C, H, W)
print(f'GAN probe images: {probe_imgs.shape}', flush=True)

# ── 2. Build trigger patterns ─────────────────────────────────────────────────
img_shape = probe_imgs.shape[1:]  # (C, H, W)
C, H, W = img_shape

# Checker: ±amplitude alternating at every pixel
rows = np.arange(H).reshape(H, 1)
cols = np.arange(W).reshape(1, W)
checker_hw = ((rows + cols) % 2).astype(np.float32) * 2.0 - 1.0  # (H, W)
checker_pattern = np.stack([checker_hw * args.amplitude] * C, axis=0)  # (C, H, W)

# Noise: fixed-seed L∞-normalised random noise (same as train.py)
rng = np.random.default_rng(12345)
noise_pattern = rng.random((C, H, W)).astype(np.float32) * 2.0 - 1.0
noise_pattern = noise_pattern / (np.abs(noise_pattern).max() + 1e-9) * args.amplitude

# Patch: delta = 1.0 at trigger pixel positions, 0 elsewhere.
# Use get_pattern_mask to extract the positions from cifar_params.yaml.
cpu_dev = torch.device('cpu')
mask = train_mod.get_pattern_mask(params, 0, img_shape, cpu_dev).numpy()  # (C, H, W) uint8
patch_pattern = mask.astype(np.float32)  # 1.0 at pattern pixels, 0 elsewhere
# The current trigger replaces the pixel with 1.0, so the effective delta
# relative to a zero-mean image is just the mask itself at amplitude 1.0.

signals = {
    'gan_probe':  probe_imgs,                         # (N, C, H, W) — averaged
    'checker':    checker_pattern[np.newaxis],         # (1, C, H, W)
    'noise':      noise_pattern[np.newaxis],
    'patch':      patch_pattern[np.newaxis],
}


# ── 3. Radially-averaged power spectrum ──────────────────────────────────────
def radial_psd(images):
    """Mean radial power spectrum over a batch of (N, C, H, W) images.

    Returns (freq_bins, power) where freq_bins are normalised to [0, 1]
    (1 = Nyquist corner of the 2D DFT, sqrt(2)/2 for axis-aligned Nyquist).
    """
    N, C, H, W = images.shape
    # Compute 2D FFT magnitude² for each image and channel, average
    power_sum = np.zeros((H, W))
    for img in images:
        for ch in range(C):
            f = np.fft.fft2(img[ch])
            power_sum += np.abs(np.fft.fftshift(f)) ** 2
    power_mean = power_sum / (N * C)

    # Build radial frequency grid (normalised: k_max = sqrt((H/2)^2 + (W/2)^2))
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H  # pixel-wavenumber, centred
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_max = np.sqrt((H / 2) ** 2 + (W / 2) ** 2)  # Nyquist corner
    K_norm = K / k_max  # normalised ∈ [0, 1]

    # Bin into 50 radial shells
    n_bins = 50
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_power = np.zeros(n_bins)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask_bin = (K_norm >= lo) & (K_norm < hi)
        bin_power[i] = power_mean[mask_bin].sum() if mask_bin.any() else 0.0

    total = bin_power.sum() + 1e-30
    return bin_centers, bin_power / total  # normalised power fractions


# ── 4. Report ──────────────────────────────────────────────────────────────────
results = {}
print('\n=== Radial power spectrum (normalised frequency, 0=DC, 1=Nyquist corner) ===')
print(f"{'signal':>12}  {'low<0.25':>10}  {'mid0.25-0.5':>12}  "
      f"{'hi>0.5':>8}  {'median_f':>9}")
print('-' * 60)

for name, imgs in signals.items():
    freqs, power = radial_psd(imgs)
    cum = np.cumsum(power)
    # Interpolate cumulative power at thresholds
    def cum_at(thresh):
        idx = np.searchsorted(freqs, thresh)
        return float(cum[min(idx, len(cum)-1)])

    low  = cum_at(0.25)
    mid  = cum_at(0.50) - low
    high = 1.0 - cum_at(0.50)
    # Median frequency: where cumulative power = 0.5
    med_idx = np.searchsorted(cum, 0.5)
    med_f = float(freqs[min(med_idx, len(freqs)-1)])

    results[name] = dict(low=low, mid=mid, high=high, median_f=med_f,
                         freqs=freqs.tolist(), power=power.tolist())
    print(f'{name:>12}  {low:>10.3f}  {mid:>12.3f}  {high:>8.3f}  {med_f:>9.4f}')

# Band-by-band detail for GAN vs checker
print('\n=== Cumulative power fraction at each frequency band (GAN vs checker) ===')
print(f"{'freq_norm':>10}  {'GAN':>8}  {'checker':>9}  {'noise':>7}  {'patch':>7}")
print('-' * 48)
gan_freqs = results['gan_probe']['freqs']
gan_cum   = np.cumsum(results['gan_probe']['power'])
chk_cum   = np.cumsum(results['checker']['power'])
nse_cum   = np.cumsum(results['noise']['power'])
pat_cum   = np.cumsum(results['patch']['power'])
for step in np.linspace(0.1, 1.0, 10):
    idx = np.searchsorted(np.array(gan_freqs), step)
    idx = min(idx, len(gan_cum) - 1)
    print(f'{step:>10.2f}  {gan_cum[idx]:>8.4f}  {chk_cum[idx]:>9.4f}  '
          f'{nse_cum[idx]:>7.4f}  {pat_cum[idx]:>7.4f}')

# Save raw arrays for potential plotting
import csv
csv_path = os.path.join(args.out, 'check11_spectral.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['signal', 'freq_norm', 'power_frac', 'cum_power'])
    for name in signals:
        freqs_arr = np.array(results[name]['freqs'])
        power_arr = np.array(results[name]['power'])
        cum_arr   = np.cumsum(power_arr)
        for fq, pw, cp in zip(freqs_arr, power_arr, cum_arr):
            wr.writerow([name, f'{fq:.5f}', f'{pw:.6f}', f'{cp:.6f}'])
print(f'\nSaved to {csv_path}')
