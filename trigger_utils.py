"""trigger_utils.py — shared trigger module.

All triggers are applied in raw [0,1] pixel space:
  x_raw → x_raw + δ → clamp([0,1]) → normalize
or for patch:
  x_raw[mask] = raw_value → normalize

Import:
    import trigger_utils as tu  (after sys.path.insert(0, FilterFL_dir))

Required API used across scripts:
    tu.VERSION            — str
    tu.MEAN, tu.STD       — per-channel lists (ImageNet stats, float32)
    tu.make_raw_delta(name, amplitude, seed)  → (C,H,W) float32 numpy
    tu.apply_additive_norm(imgs_norm, delta_raw_np) → tensor (same device)
    tu.apply_patch_norm(imgs_norm, patch_mask, raw_value=1.0)  → tensor
    tu.get_patch_mask(positions, device) → (C,H,W) bool tensor
"""

import numpy as np
import torch

VERSION = "1.3.0"

# ImageNet normalisation used by FilterFL for CIFAR-10 (handle.py lines 69-73).
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# --- internal helpers -------------------------------------------------------

_MEAN_T = {}   # device → (1,3,1,1) tensor
_STD_T  = {}

def _tensors(device):
    dev_key = str(device)
    if dev_key not in _MEAN_T:
        _MEAN_T[dev_key] = torch.tensor(MEAN, dtype=torch.float32,
                                        device=device).view(1, 3, 1, 1)
        _STD_T[dev_key]  = torch.tensor(STD,  dtype=torch.float32,
                                        device=device).view(1, 3, 1, 1)
    return _MEAN_T[dev_key], _STD_T[dev_key]


def _to_raw(x_norm):
    """Normalized (N,3,H,W) → raw [0,1] (no clamp; caller clamps if needed)."""
    m, s = _tensors(x_norm.device)
    return x_norm * s + m


def _to_norm(x_raw):
    """Raw [0,1] (N,3,H,W) → normalized."""
    m, s = _tensors(x_raw.device)
    return (x_raw - m) / s


# --- public API -------------------------------------------------------------

def make_raw_delta(name, amplitude=0.1145, seed=42, C=3, H=32, W=32):
    """Return (C,H,W) float32 numpy delta in raw [0,1] space.

    name:
      'checker'                — ±amplitude alternating checkerboard (all channels)
      'checker_energy_matched' — checker at amp=3.268/√(C·H·W)=0.05896 (ignores amplitude)
      'eps4'                   — ±4/255 checkerboard (amplitude ignored)
      'eps8'                   — ±8/255 checkerboard (amplitude ignored)
      'eps16'                  — ±16/255 checkerboard (amplitude ignored)
      'eps32'                  — ±32/255 checkerboard (amplitude ignored)
      'eps64'                  — ±64/255 checkerboard (amplitude ignored)
      'noise'                  — ±amplitude fixed-seed uniform random (seed=42 default)
      'stripes'                — ±amplitude alternating horizontal rows (all channels)
      'sin16'                  — sinusoidal period-16 scaled to match checker L2
      'sin32'                  — sinusoidal period-32 scaled to match checker L2
      'corner4x4'              — ±amplitude checkerboard in top-left 4×4 pixels
      'corner4x4_energy_matched' — corner4x4 at amp=3.268/√(C·16)=0.4717 (ignores amplitude)
      'cov64pos'               — ±amplitude at 64 deterministic positions (seed controls selection)
      'gradient'               — horizontal centered linear ramp, hardcoded amp=0.2 (L2_raw≈3.3)
      'sin_large'              — sin16 at 3× checker_energy_matched amplitude (L2_raw≈9.8)

    Energy-matched amplitudes reference patch L2_raw=3.268 (unit-test batch, 500 CIFAR-10
    test images from pickle). Pooling-gain batch gives 3.384 (DataLoader, different composition).
    """
    rows = np.arange(H, dtype=np.float32).reshape(H, 1)
    cols = np.arange(W, dtype=np.float32).reshape(1, W)

    if name == 'checker':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * amplitude] * C, axis=0)  # (C,H,W)

    elif name == 'checker_energy_matched':
        _em_amp = 3.268 / (C * H * W) ** 0.5   # 0.05896; ignores amplitude param
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * _em_amp] * C, axis=0)

    elif name == 'eps4':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * (4.0 / 255.0)] * C, axis=0)

    elif name == 'eps8':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * (8.0 / 255.0)] * C, axis=0)

    elif name == 'eps16':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * (16.0 / 255.0)] * C, axis=0)

    elif name == 'eps32':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * (32.0 / 255.0)] * C, axis=0)

    elif name == 'eps64':
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        delta = np.stack([hw * (64.0 / 255.0)] * C, axis=0)

    elif name == 'noise':
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=(C, H, W)).astype(np.float32)
        delta = signs * amplitude

    elif name == 'stripes':
        row_sign = ((np.arange(H, dtype=int) % 2) * 2 - 1).astype(np.float32)  # ±1 per row
        pat = np.broadcast_to(row_sign.reshape(H, 1), (H, W)) * amplitude
        delta = np.stack([pat.copy()] * C, axis=0)

    elif name == 'sin16':
        row_v = np.sin(2 * np.pi / 16 * np.arange(H)).astype(np.float32)
        ss = float(np.sum(row_v ** 2))
        sin_amp = float(amplitude * np.sqrt(H / ss))  # scale to match checker L2
        row_scaled = (row_v * sin_amp)
        delta = np.stack([np.broadcast_to(row_scaled.reshape(H, 1), (H, W)).copy()] * C, axis=0)

    elif name == 'sin32':
        row_v = np.sin(2 * np.pi / 32 * np.arange(H)).astype(np.float32)
        ss = float(np.sum(row_v ** 2))
        sin_amp = float(amplitude * np.sqrt(H / ss))  # scale to match checker L2
        row_scaled = (row_v * sin_amp)
        delta = np.stack([np.broadcast_to(row_scaled.reshape(H, 1), (H, W)).copy()] * C, axis=0)

    elif name == 'corner4x4':
        delta = np.zeros((C, H, W), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                delta[:, r, c] = (((r + c) % 2) * 2.0 - 1.0) * amplitude

    elif name == 'corner4x4_energy_matched':
        _em_amp_c4 = 3.268 / (C * 16) ** 0.5   # 0.4717; ignores amplitude param
        delta = np.zeros((C, H, W), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                delta[:, r, c] = (((r + c) % 2) * 2.0 - 1.0) * _em_amp_c4

    elif name == 'blend_87':
        # 87.5% checker + 12.5% stripes, renormalized to checker L2.
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        d_chk = np.stack([hw * amplitude] * C, axis=0)
        row_sign = ((np.arange(H, dtype=int) % 2) * 2 - 1).astype(np.float32)
        pat = np.broadcast_to(row_sign.reshape(H, 1), (H, W)) * amplitude
        d_str = np.stack([pat.copy()] * C, axis=0)
        d_mix = 0.875 * d_chk + 0.125 * d_str
        l2_chk = float(np.linalg.norm(d_chk))
        l2_mix = float(np.linalg.norm(d_mix))
        delta = d_mix * (l2_chk / l2_mix) if l2_mix > 1e-9 else d_mix

    elif name == 'blend_75':
        # 75% checker + 25% stripes, renormalized to checker L2.
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        d_chk = np.stack([hw * amplitude] * C, axis=0)
        row_sign = ((np.arange(H, dtype=int) % 2) * 2 - 1).astype(np.float32)
        pat = np.broadcast_to(row_sign.reshape(H, 1), (H, W)) * amplitude
        d_str = np.stack([pat.copy()] * C, axis=0)
        d_mix = 0.75 * d_chk + 0.25 * d_str
        l2_chk = float(np.linalg.norm(d_chk))
        l2_mix = float(np.linalg.norm(d_mix))
        delta = d_mix * (l2_chk / l2_mix) if l2_mix > 1e-9 else d_mix

    elif name == 'blend_25':
        # 25% checker + 75% stripes, renormalized to checker L2.
        hw = ((rows.astype(int) + cols.astype(int)) % 2).astype(np.float32) * 2.0 - 1.0
        d_chk = np.stack([hw * amplitude] * C, axis=0)
        row_sign = ((np.arange(H, dtype=int) % 2) * 2 - 1).astype(np.float32)
        pat = np.broadcast_to(row_sign.reshape(H, 1), (H, W)) * amplitude
        d_str = np.stack([pat.copy()] * C, axis=0)
        d_mix = 0.25 * d_chk + 0.75 * d_str
        l2_chk = float(np.linalg.norm(d_chk))
        l2_mix = float(np.linalg.norm(d_mix))
        delta = d_mix * (l2_chk / l2_mix) if l2_mix > 1e-9 else d_mix

    elif name == 'stripes_p4':
        # Period-4 horizontal stripes: ±amplitude alternating per 2-row block.
        row_sign = ((np.arange(H, dtype=int) // 2) % 2 * 2 - 1).astype(np.float32)
        pat = np.broadcast_to(row_sign.reshape(H, 1), (H, W)) * amplitude
        delta = np.stack([pat.copy()] * C, axis=0)

    elif name == 'low_col_opt':
        # Gradient-optimised trigger: minimises max |cos(feat_shift_dir, W_fc_norm[c])|
        # on the FL-trained clean model. Constrained to checker L2.
        _lco_path = ('/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl'
                     '/ffl_step5/low_col_opt_delta.npy')
        delta = np.load(_lco_path).astype(np.float32)

    elif name == 'truck_grad':
        # Input-gradient trigger: mean_x grad_x(max_{c!=0} logit_c(x)) on clean ResNet8.
        # Maximises the natural-class margin Δ_nat; pattern is non-grating noise.
        # Pre-computed from step5_clean_resnet8.pt, normalized to checker L2.
        _tg_path = ('/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl'
                    '/ffl_step5/truck_grad_delta.npy')
        delta = np.load(_tg_path).astype(np.float32)

    elif name == 'cov64pos':
        # 64 deterministic random positions; ±amplitude alternating by position index.
        rng = np.random.RandomState(seed)
        positions = [(int(r), int(c))
                     for r, c in zip(rng.randint(0, H, 64), rng.randint(0, W, 64))]
        delta = np.zeros((C, H, W), dtype=np.float32)
        for k, (r, c) in enumerate(positions):
            sign = 1.0 if k % 2 == 0 else -1.0
            delta[:, r, c] = sign * amplitude

    elif name == 'gradient':
        # Horizontal centered linear ramp: v_col = 0.2*(col/(W-1) - 0.5), range≈(−0.10, +0.10).
        # Hardcoded amp=0.2; empirical L2_raw≈3.3 (close to reference 3.268). Ignores amplitude.
        _grad_amp = 0.2
        w_frac = np.arange(W, dtype=np.float32) / max(W - 1, 1)
        pat = (w_frac - 0.5) * _grad_amp             # (W,), centered at 0
        pat2d = np.broadcast_to(pat.reshape(1, W), (H, W)).copy()
        delta = np.stack([pat2d] * C, axis=0)

    elif name == 'sin_large':
        # sin16 at 3× checker_energy_matched amplitude ≈ 0.1769; L2_raw≈9.8 (3× reference).
        # Ignores amplitude param. Tests whether high-amplitude low-frequency trigger implants.
        _sl_amp = 3.0 * (3.268 / (C * H * W) ** 0.5)   # 3 × 0.05896 = 0.1769
        row_v = np.sin(2 * np.pi / 16 * np.arange(H)).astype(np.float32)
        ss = float(np.sum(row_v ** 2))
        sin_amp = float(_sl_amp * np.sqrt(H / ss))
        row_scaled = row_v * sin_amp
        delta = np.stack([np.broadcast_to(row_scaled.reshape(H, 1), (H, W)).copy()] * C, axis=0)

    else:
        raise ValueError(f"Unknown trigger name: {name!r}")

    return delta.astype(np.float32)


def apply_additive_norm(imgs_norm, delta_raw_np):
    """Apply additive trigger in raw space; return normalized tensor (same device).

    imgs_norm   : (N,3,H,W) normalized tensor
    delta_raw_np: (3,H,W) numpy array in raw [0,1] space
    """
    delta_t = torch.from_numpy(delta_raw_np).to(imgs_norm.device).unsqueeze(0)
    raw = _to_raw(imgs_norm)
    raw_triggered = torch.clamp(raw + delta_t, 0.0, 1.0)
    return _to_norm(raw_triggered)


def apply_patch_norm(imgs_norm, patch_mask, raw_value=1.0):
    """Apply patch (pixel replacement) in raw space; return normalized tensor.

    imgs_norm  : (N,3,H,W) normalized tensor
    patch_mask : (3,H,W) bool tensor
    raw_value  : replacement pixel value in [0,1] raw space (default 1.0 = white)
    """
    raw = _to_raw(imgs_norm)
    fill = torch.full_like(raw, raw_value)
    mask_exp = patch_mask.unsqueeze(0).expand_as(raw)
    raw_triggered = torch.where(mask_exp, fill, raw)
    # No clamp needed: raw_value must be in [0,1] and raw is already in [0,1].
    return _to_norm(raw_triggered)


def checker_amplitude_for_l2(target_l2, C=3, H=32, W=32):
    """Checker amplitude that achieves target_l2 on unclamped images.

    For a full-image checkerboard at ±amp (all C×H×W pixels hit):
        L2_raw = amp × sqrt(C × H × W)
    Inverting gives the amplitude needed to match a target L2.
    Note: clamping will reduce the empirical L2 slightly, so this is an upper bound.
    """
    return target_l2 / (C * H * W) ** 0.5


def get_patch_mask(positions, device):
    """Build (3,H,W) bool tensor from list of [row,col] positions (CIFAR-10: 32×32)."""
    C, H, W = 3, 32, 32
    mask = torch.zeros(C, H, W, dtype=torch.bool, device=device)
    for pos in positions:
        mask[:, int(pos[0]), int(pos[1])] = True
    return mask
