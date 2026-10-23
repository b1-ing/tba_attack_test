"""Probe spectrum analysis for the G-vs-H experiment.

For each saved round, measures:
  - pixel_probe checker-band power fraction  (2D FFT, bin at Nyquist corner)
  - pixel_probe high-freq power fraction     (radial |f| > 0.4)
  - GAN reference probe (same round's model states, re-run gan_defense)
  - checker trigger ceiling
  - random-noise ceiling (sanity check: should be uniform across all freqs)

Usage:
    python3 step5_probe_spectrum_analysis.py <probe_save_dir> <filterfl_dir>

<probe_save_dir> must contain:
    pixel_probe_ep{NNNN}.pt   (N, 3, 32, 32) probe tensors
    model_ep{NNNN}.pt         {'d1': state_dict, 'd2': state_dict}
"""

import sys, os, glob, re
import numpy as np
import torch

probe_dir = sys.argv[1]
ffl_dir   = sys.argv[2]

sys.path.insert(0, ffl_dir)
import trigger_utils as tu
from gan import gan_defense

# Minimal params for gan_defense (only fields it inspects)
GAN_PARAMS = {'type': 'CIFAR10', 'gan_epochs1': 10, 'gan_epochs2': 5}


def probe_spectrum(batch_t):
    """Compute checker-band and high-freq power fractions from a probe batch.

    batch_t: (N, C, H, W) tensor or numpy, in any linear-scale space.
    Averages over N and C before computing 2D FFT (one representative image).
    Returns dict: checker_frac, high_frac.
    """
    x = np.array(batch_t, dtype=np.float64)
    img = x.mean(axis=(0, 1))          # (H, W) — average over batch and channels
    H, W = img.shape

    F = np.fft.fft2(img)               # (H, W) complex, unshifted
    power = np.abs(F) ** 2
    total = power.sum() + 1e-30

    # Checker bin: pure ±1 checkerboard concentrates ALL energy at (H//2, W//2)
    checker_frac = power[H // 2, W // 2] / total

    # High-freq: radial normalized frequency > 0.4 (0.5 = Nyquist corner)
    kx = np.arange(W, dtype=float)
    ky = np.arange(H, dtype=float)
    kx_c = np.where(kx < W / 2, kx, kx - W) / W   # centered, in [−0.5, 0.5)
    ky_c = np.where(ky < H / 2, ky, ky - H) / H
    freq2d = np.sqrt(kx_c[np.newaxis, :] ** 2 + ky_c[:, np.newaxis] ** 2)
    high_frac = power[freq2d > 0.4].sum() / total

    return {'checker_frac': float(checker_frac), 'high_frac': float(high_frac)}


# --- ceiling: checker trigger (raw space, amplitude 0.1145) ----------------
checker_delta = tu.make_raw_delta('checker')      # (3, 32, 32)
ceil_checker  = probe_spectrum(checker_delta[np.newaxis])

# --- sanity: noise trigger (random signs, same amplitude) ------------------
noise_delta = tu.make_raw_delta('noise')
ceil_noise = probe_spectrum(noise_delta[np.newaxis])

print("Reference spectra:")
print(f"  checker trigger (ceiling): checker_frac={ceil_checker['checker_frac']:.4f}  "
      f"high_frac={ceil_checker['high_frac']:.4f}")
print(f"  noise trigger  (sanity):   checker_frac={ceil_noise['checker_frac']:.4f}  "
      f"high_frac={ceil_noise['high_frac']:.4f}")
print()

# --- per-round analysis -----------------------------------------------------
pt_files = sorted(glob.glob(os.path.join(probe_dir, 'pixel_probe_ep*.pt')))
if not pt_files:
    print(f"ERROR: no pixel_probe_ep*.pt files found in {probe_dir}")
    sys.exit(1)

rows = []
for pf in pt_files:
    ep = int(re.search(r'ep(\d+)', os.path.basename(pf)).group(1))
    probe_t = torch.load(pf, map_location='cpu', weights_only=True)
    pix_spec = probe_spectrum(probe_t.numpy())

    # GAN reference at same model state
    mf = os.path.join(probe_dir, f'model_ep{ep:04d}.pt')
    gan_spec = None
    if os.path.exists(mf):
        m = torch.load(mf, map_location='cpu', weights_only=False)
        d1, d2 = m['d1'], m['d2']
        try:
            gan_probes = gan_defense(d1, d2, GAN_PARAMS, ep)
            gp = gan_probes[0]
            if isinstance(gp, torch.Tensor):
                gp = gp.detach().cpu().numpy()
            else:
                gp = np.array(gp)
            gan_spec = probe_spectrum(gp)
        except Exception as e:
            print(f"  ep={ep}: gan_defense failed: {e}")

    rows.append((ep, pix_spec, gan_spec))

# --- print table ------------------------------------------------------------
print("Probe spectrum over time — checker arm, pixel probes vs GAN reference")
print(f"{'Round':>6}  {'phase':>12}  "
      f"{'PIX chk_frac':>13}  {'PIX hi_frac':>12}  "
      f"{'GAN chk_frac':>13}  {'GAN hi_frac':>12}")
print('-' * 80)

for ep, pix, gan in rows:
    phase = 'pre-trans' if ep <= 339 else 'post-trans'
    g_chk = f"{gan['checker_frac']:.4f}" if gan else '—'
    g_hi  = f"{gan['high_frac']:.4f}"  if gan else '—'
    print(f"  {ep:4d}  {phase:>12}  "
          f"{pix['checker_frac']:>13.4f}  {pix['high_frac']:>12.4f}  "
          f"{g_chk:>13}  {g_hi:>12}")

print()
print(f"  checker ceiling: chk_frac={ceil_checker['checker_frac']:.4f}  "
      f"hi_frac={ceil_checker['high_frac']:.4f}")
print()
print("Interpretation:")
print("  chk_frac ~ ceiling → pixel probes encode Nyquist band (link G refuted if S3 still decays)")
print("  chk_frac ~ GAN floor → optimizer converged to smooth probes (link G-ish mechanism)")
