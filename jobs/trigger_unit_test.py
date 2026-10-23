#!/usr/bin/env python3
"""Unit tests for trigger_utils.py — no GPU required.

Reports per variant:
  L2_raw   : mean per-image L2 norm of (triggered_raw - clean_raw)
  Linf_raw : max per-pixel absolute delta in raw [0,1] space
  clamp_pct: percentage of pixel-channel slots affected by [0,1] clamping
  all_valid : True if all triggered pixels are in [0,1] raw

Also checks:
  eps8/eps4 L∞ ratio == 2.00 (exact)
  Patch pixels set to 1.0 raw (white)
  Patch L2_raw (for energy-matched checker amplitude recomputation)
"""

import sys
import os
import pickle
import numpy as np

# ── locate trigger_utils ──────────────────────────────────────────────────────
REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE  = os.path.join(REPO, 'FilterFL')
sys.path.insert(0, CODE)

import trigger_utils as tu
print(f'trigger_utils VERSION={tu.VERSION}')
print(f'MEAN={tu.MEAN}')
print(f'STD={tu.STD}')

# ── load CIFAR-10 test images (raw [0,1]) ────────────────────────────────────
CIFAR_PATH = os.path.expanduser(
    '~/filterfl_test/filterfl_test/data/cifar-10-batches-py/test_batch')

with open(CIFAR_PATH, 'rb') as f:
    d = pickle.load(f, encoding='bytes')

raw_all = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels  = np.array(d[b'labels'])

N = 500
raw  = raw_all[:N]          # (500, 3, 32, 32) raw [0,1]
mean = np.array(tu.MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
std  = np.array(tu.STD,  dtype=np.float32).reshape(1, 3, 1, 1)
norm = (raw - mean) / std   # (500, 3, 32, 32) normalized

print(f'\nLoaded {N} CIFAR-10 test images (first {N} rows).')
print(f'Raw range: [{raw.min():.4f}, {raw.max():.4f}]')

# ── patch mask (from cifar_params.yaml default) ───────────────────────────────
PATCH_POSITIONS = [
    [0,0],[0,1],[0,2],[0,3],[0,6],[0,7],[0,8],[0,9],
    [3,0],[3,1],[3,2],[3,3],[3,6],[3,7],[3,8],[3,9],
]
N_PATCH_PX = len(PATCH_POSITIONS)  # per channel


def apply_trigger_np(name, raw_imgs, amplitude=0.1145):
    """Apply trigger in numpy raw [0,1] space → return triggered raw (N,3,32,32)."""
    if name == 'patch':
        out = raw_imgs.copy()
        for pos in PATCH_POSITIONS:
            out[:, :, pos[0], pos[1]] = 1.0
        return out
    else:
        delta = tu.make_raw_delta(name, amplitude=amplitude, seed=7 if name == 'noise' else 42)
        triggered = raw_imgs + delta[np.newaxis]
        return np.clip(triggered, 0.0, 1.0)


def stats(name, raw_clean, amplitude=0.1145):
    raw_t = apply_trigger_np(name, raw_clean, amplitude=amplitude)
    delta_raw = raw_t - raw_clean          # (N, 3, 32, 32)

    # L2_raw: per-image, then mean
    l2 = np.sqrt((delta_raw ** 2).sum(axis=(1, 2, 3))).mean()

    # L∞_raw: absolute max over all images/pixels
    linf = np.abs(delta_raw).max()

    # clamp fraction: pixels that would have exceeded [0,1] without clamp
    pre_clamp = raw_clean + (tu.make_raw_delta(name, amplitude=amplitude, seed=7 if name == 'noise' else 42)[np.newaxis]
                              if name != 'patch' else np.zeros_like(raw_clean))
    if name != 'patch':
        clamped = ((pre_clamp > 1.0) | (pre_clamp < 0.0)).mean()
    else:
        clamped = 0.0  # patch sets to exact value, no overflow

    # validity: all triggered pixels in [0,1]
    valid = bool((raw_t >= 0.0).all() and (raw_t <= 1.0).all())

    return l2, linf, clamped * 100.0, valid


# ── run tests ─────────────────────────────────────────────────────────────────
AMP = 0.1145
H_IMG, W_IMG = 32, 32

# ── Additional trigger types (needed for pooling gain / Step 3) ───────────────
def _sin_amp_raw(T):
    """Energy-matched sinusoidal amplitude in raw space (same L2_raw as checker)."""
    ss = np.sum(np.sin(2 * np.pi / T * np.arange(H_IMG)) ** 2)
    return AMP * np.sqrt(H_IMG / ss)


def apply_additive_np(delta, raw_imgs):
    """Apply (C,H,W) numpy delta to (N,C,H,W) raw images, clamp [0,1]."""
    pre = raw_imgs + delta[np.newaxis]
    clamped = ((pre > 1.0) | (pre < 0.0)).mean() * 100.0
    return np.clip(pre, 0.0, 1.0), clamped


def spectral_power_below(delta, f_thresh=0.3):
    """Fraction of sinusoidal-axis spectral power at spatial freq < f_thresh.
    delta: (C,H,W) numpy.  Averages over C, analyses row dimension (axis=1).
    f=0 to 0.5 normalized; DC excluded.
    """
    row_means = delta.mean(axis=(0, 2))  # (H,) — mean over channels & cols
    F = np.fft.rfft(row_means)[1:]       # drop DC
    power = np.abs(F) ** 2
    freqs = np.fft.rfftfreq(H_IMG)[1:]
    low_mask = freqs < f_thresh
    frac = power[low_mask].sum() / (power.sum() + 1e-30)
    return frac


def make_delta_np(name, amp=AMP):
    """Build numpy (3,H,W) raw-space delta for extended trigger types."""
    rows = np.arange(H_IMG, dtype=np.float32).reshape(H_IMG, 1)
    cols = np.arange(W_IMG, dtype=np.float32).reshape(1, W_IMG)
    hw   = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0

    if name == 'stripes':
        row_v = (np.arange(H_IMG) % 2).astype(np.float32) * 2.0 - 1.0
        pat = np.broadcast_to(row_v.reshape(H_IMG, 1), (H_IMG, W_IMG)) * amp
        return np.stack([pat] * 3, axis=0).astype(np.float32)

    elif name == 'corner4x4':
        delta = np.zeros((3, H_IMG, W_IMG), np.float32)
        for r in range(4):
            for c in range(4):
                delta[:, r, c] = ((r + c) % 2) * 2.0 - 1.0
        return (delta * amp).astype(np.float32)

    elif name == 'sin16':
        a = _sin_amp_raw(16)
        row_v = np.sin(2 * np.pi / 16 * np.arange(H_IMG)).astype(np.float32) * a
        pat = np.broadcast_to(row_v.reshape(H_IMG, 1), (H_IMG, W_IMG))
        return np.stack([pat] * 3, axis=0).astype(np.float32)

    elif name == 'sin32':
        a = _sin_amp_raw(32)
        row_v = np.sin(2 * np.pi / 32 * np.arange(H_IMG)).astype(np.float32) * a
        pat = np.broadcast_to(row_v.reshape(H_IMG, 1), (H_IMG, W_IMG))
        return np.stack([pat] * 3, axis=0).astype(np.float32)

    else:
        raise ValueError(name)


# Compute energy-matched amplitude from patch L2 (needed before extended table)
def _patch_l2(raw_imgs):
    raw_t = raw_imgs.copy()
    for pos in PATCH_POSITIONS:
        raw_t[:, :, pos[0], pos[1]] = 1.0
    return np.sqrt(((raw_t - raw_imgs) ** 2).sum(axis=(1, 2, 3))).mean()


# ── PASS 1: core variants ─────────────────────────────────────────────────────
CORE_VARIANTS = [
    ('checker',  AMP),
    ('eps4',     AMP),
    ('eps8',     AMP),
    ('noise',    AMP),
    ('patch',    AMP),
]

print(f'\n{"trigger":>22}  {"L2_raw":>8}  {"Linf_raw":>10}  {"clamp%":>8}  {"valid":>6}  {"pow<0.3":>8}')
print('-' * 72)

results = {}
for name, amp in CORE_VARIANTS:
    l2, linf, clamp_pct, valid = stats(name, raw, amplitude=amp)
    delta_np = (tu.make_raw_delta(name, amplitude=amp, seed=7 if name == 'noise' else 42)
                if name != 'patch' else None)
    spow = spectral_power_below(delta_np) if delta_np is not None else float('nan')
    results[name] = dict(l2=l2, linf=linf, clamp_pct=clamp_pct, valid=valid, spow=spow)
    spow_str = f'{spow:.3f}' if not np.isnan(spow) else '  —'
    print(f'{name:>22}  {l2:>8.5f}  {linf:>10.6f}  {clamp_pct:>7.3f}%  {"OK" if valid else "FAIL":>6}  {spow_str:>8}')

# ── ratio check ───────────────────────────────────────────────────────────────
print()
ratio = results['eps8']['linf'] / results['eps4']['linf']
eps8_ok = abs(ratio - 2.0) < 1e-4
print(f'eps8/eps4 L∞ ratio = {ratio:.6f}  {"PASS" if eps8_ok else "FAIL (expected 2.00)"}')

# ── patch pixel check ─────────────────────────────────────────────────────────
raw_t = apply_trigger_np('patch', raw)
for pos in PATCH_POSITIONS:
    assert (raw_t[:, :, pos[0], pos[1]] == 1.0).all(), f'Patch pixel {pos} not 1.0 raw!'
print('Patch pixels are all 1.0 raw: PASS')

# ── patch L2_raw and energy-matched amplitude ─────────────────────────────────
patch_l2 = results['patch']['l2']
em_amp   = tu.checker_amplitude_for_l2(patch_l2)
print(f'\nPatch L2_raw (mean over {N} images) = {patch_l2:.5f}')
print(f'Energy-matched checker amplitude    = {em_amp:.5f}  '
      f'(= {patch_l2:.5f} / sqrt({3*32*32}) = {patch_l2:.5f} / {(3*32*32)**0.5:.2f})')

# ── PASS 2: extended variants (Step 3 pre-req, not blocking reruns) ───────────
patch_l2_exact = _patch_l2(raw)
em_amp_exact   = tu.checker_amplitude_for_l2(patch_l2_exact)

EXTENDED = [
    ('stripes',                  make_delta_np('stripes')),
    ('corner4x4',                make_delta_np('corner4x4')),
    ('sin16',                    make_delta_np('sin16')),
    ('sin32',                    make_delta_np('sin32')),
    ('checker_energy_matched',   tu.make_raw_delta('checker_energy_matched')),
    ('corner4x4_energy_matched', tu.make_raw_delta('corner4x4_energy_matched')),
    ('gradient',                 tu.make_raw_delta('gradient')),
    ('sin_large',                tu.make_raw_delta('sin_large')),
]

print(f'\n{"trigger":>22}  {"L2_raw":>8}  {"Linf_raw":>10}  {"clamp%":>8}  {"valid":>6}  {"pow<0.3":>8}')
print('-' * 72)

for name, delta_np in EXTENDED:
    triggered, clamp_pct = apply_additive_np(delta_np, raw)
    delta_raw = triggered - raw
    l2    = np.sqrt((delta_raw ** 2).sum(axis=(1, 2, 3))).mean()
    linf  = np.abs(delta_raw).max()
    valid = bool((triggered >= 0.0).all() and (triggered <= 1.0).all())
    spow  = spectral_power_below(delta_np)
    results[name] = dict(l2=l2, linf=linf, clamp_pct=clamp_pct, valid=valid, spow=spow)
    print(f'{name:>22}  {l2:>8.5f}  {linf:>10.6f}  {clamp_pct:>7.3f}%  {"OK" if valid else "FAIL":>6}  {spow:.3f}')

# ── validity summary ──────────────────────────────────────────────────────────
print()
all_pass = all(r['valid'] for r in results.values()) and eps8_ok
print(f'All checks {"PASSED" if all_pass else "FAILED — see above"}.')
