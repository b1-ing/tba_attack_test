#!/usr/bin/env python3
"""Subspace overlap: does the adversarial update lie in the span of benign updates?

For each saved round (client_deltas_ep{N}.pt):
  1. Flatten all parameter deltas to vectors (excluding int64 params)
  2. Stack benign deltas into matrix B (n_ben × D)
  3. Compute SVD of B; take first k singular vectors (k = min(n_ben, rank))
  4. Project Δw_adv onto span(B): proj = U U^T Δw_adv
  5. overlap_frac = ||proj||^2 / ||Δw_adv||^2

Also reports:
  - ||Δw_adv|| relative to mean ||Δw_ben|| (scale ratio)
  - Per-round s3 / margin from the log (if log path provided)

Interpretation:
  overlap ≈ 0 → adv update is orthogonal to benign span (distinguishable in weight space)
  overlap ≈ 1 → adv update lies inside benign span (hidden from weight-space inspection)
"""

import argparse, os, sys, re
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--delta-dir',  required=True,  help='Dir containing client_deltas_ep*.pt')
ap.add_argument('--log',        default=None,   help='Training log for per-round s3 context')
ap.add_argument('--rounds',     nargs='+', type=int, default=None,
                help='Specific rounds to analyse (default: all saved)')
ap.add_argument('--out',        required=True)
args = ap.parse_args()


def flatten_delta(delta_dict):
    """Concatenate all float parameter deltas into one 1-D numpy vector."""
    parts = []
    for k in sorted(delta_dict.keys()):
        t = delta_dict[k]
        if t.dtype == torch.int64:
            continue
        parts.append(t.float().cpu().numpy().ravel())
    return np.concatenate(parts)


def subspace_overlap(adv_vec, ben_vecs):
    """Fraction of adv_vec variance explained by span(ben_vecs).

    Returns (overlap_frac, n_components_used).
    """
    if len(ben_vecs) == 0:
        return float('nan'), 0
    B = np.stack(ben_vecs, axis=0)        # (n_ben, D)
    # SVD for numerical stability; use thin SVD
    try:
        U, s, Vt = np.linalg.svd(B, full_matrices=False)  # U: (n_ben, n_ben), Vt: (n_ben, D)
    except np.linalg.LinAlgError:
        return float('nan'), 0
    # Vt rows are the right singular vectors (basis of row space of B = span of ben updates)
    # Project adv_vec onto each basis vector
    proj_sq = float(np.sum((Vt @ adv_vec) ** 2))
    adv_sq  = float(np.dot(adv_vec, adv_vec))
    if adv_sq < 1e-30:
        return float('nan'), 0
    return proj_sq / adv_sq, Vt.shape[0]


def parse_log_s3(log_path):
    """Extract per-round s3 (FEAT_ALIGN line) from training log → {epoch: s3}."""
    if log_path is None or not os.path.exists(log_path):
        return {}
    s3_map = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r'FEAT_ALIGN ep=(\d+) client=\S+ fa=[\d.-]+ s3=([\d.]+)', line)
            if m:
                ep, s3 = int(m.group(1)), float(m.group(2))
                s3_map.setdefault(ep, []).append(s3)
    return {ep: max(vs) for ep, vs in s3_map.items()}   # worst-case (max s3 among adv clients)


# ── find saved rounds ─────────────────────────────────────────────────────────
files = sorted(f for f in os.listdir(args.delta_dir)
               if f.startswith('client_deltas_ep') and f.endswith('.pt'))
if not files:
    print(f'No client_deltas_ep*.pt files in {args.delta_dir}', file=sys.stderr)
    sys.exit(1)

s3_map = parse_log_s3(args.log)

rows = []
for fname in files:
    m = re.search(r'ep(\d+)', fname)
    if not m:
        continue
    ep = int(m.group(1))
    if args.rounds and ep not in args.rounds:
        continue

    data = torch.load(os.path.join(args.delta_dir, fname), map_location='cpu', weights_only=False)
    n_adv = data.get('n_adv', len(data['adv']))

    if n_adv == 0:
        rows.append({'ep': ep, 'n_adv': 0, 'overlap': float('nan'),
                     'adv_norm': float('nan'), 'scale_ratio': float('nan'),
                     's3_max': s3_map.get(ep, float('nan'))})
        continue

    adv_vecs = [flatten_delta(d) for d in data['adv']]
    ben_vecs  = [flatten_delta(d) for d in data['ben']]

    # Use the first (or only) adversarial update
    adv_vec = adv_vecs[0]
    adv_norm = float(np.linalg.norm(adv_vec))
    ben_norms = [float(np.linalg.norm(v)) for v in ben_vecs]
    mean_ben_norm = float(np.mean(ben_norms)) if ben_norms else float('nan')
    scale_ratio = adv_norm / mean_ben_norm if mean_ben_norm > 0 else float('nan')

    overlap, n_comp = subspace_overlap(adv_vec, ben_vecs)
    rows.append({'ep': ep, 'n_adv': n_adv, 'overlap': overlap,
                 'adv_norm': adv_norm, 'scale_ratio': scale_ratio,
                 's3_max': s3_map.get(ep, float('nan')),
                 'n_comp': n_comp})

# ── write output ──────────────────────────────────────────────────────────────
header = (f"{'Rnd':>5}  {'n_adv':>5}  {'overlap':>9}  "
          f"{'adv_norm':>10}  {'scale_ratio':>11}  {'s3_max':>8}")
sep    = '-' * len(header)

lines = [header, sep]
for r in rows:
    ov  = f"{r['overlap']:.4f}"   if not np.isnan(r['overlap'])      else '     nan'
    sc  = f"{r['scale_ratio']:.3f}" if not np.isnan(r['scale_ratio']) else '      nan'
    an  = f"{r['adv_norm']:.2f}"  if not np.isnan(r['adv_norm'])     else '       nan'
    s3  = f"{r['s3_max']:.4f}"   if not np.isnan(r['s3_max'])        else '     nan'
    lines.append(f"{r['ep']:>5}  {r['n_adv']:>5}  {ov:>9}  {an:>10}  {sc:>11}  {s3:>8}")

output = '\n'.join(lines) + '\n'
print(output)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, 'w') as f:
    f.write(output)
print(f'Saved to {args.out}')
