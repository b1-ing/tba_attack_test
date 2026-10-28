#!/usr/bin/env python3
"""
Step 5 distributional separability analysis.

For each defense, tests whether the low-norm attacker is statistically
distinguishable from honest clients on the defense's own detection statistic.

Parses per-client per-round statistics from enhanced log files and computes:
  Step 1 (gate): AUROC, KS statistic, overlap % for attacker vs honest.
  Step 2 (threshold sweep, only for defenses that pass gate): ROC tradeoff.
  Step 3: summary table.

Log format expected:
  FilterFL:
    FEAT_ALIGN ep={e} client={c} fa={f} s3={s}       (adversarial)
    FEAT_ALIGN_BEN ep={e} client={c} fa={f} s3={s}   (benign, log_ben_s3=True)
  FLAME / FLTrust (all defenses):
    [CLIENT_FLAGS] ep={e} order=[c1,c2,...] adv_mask=[0,1,0,...]
  FLAME:
    FLAME ep={e} median_norm={m} norms=[n1,n2,...]
  FLTrust:
    FLTrust ep={e}: srv_norm={s} trust=[t1,t2,...] sum={s}

Usage:
  python3 step5_distrib_analysis.py \\
    --defense filterfl --log /path/to/run.log \\
    --out /path/to/output.txt [--cap 0.057] [--auroc-gate 0.70]
"""

import sys, re, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

AUROC_GATE = 0.70   # AUROC (distance from 0.5) above which defense SEPARATES attacker


def parse_filterfl(log_path):
    """Parse FEAT_ALIGN and FEAT_ALIGN_BEN lines → {ep: {client: s3}}."""
    adv_scores, ben_scores = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r'FEAT_ALIGN ep=(\d+) client=(\S+) fa=([\d.eE+-]+) s3=([\d.eE+-]+)', line)
            if m:
                adv_scores.append(float(m.group(4)))
                continue
            m = re.search(r'FEAT_ALIGN_BEN ep=(\d+) client=(\S+) fa=([\d.eE+-]+) s3=([\d.eE+-]+)', line)
            if m:
                ben_scores.append(float(m.group(4)))
    return np.array(adv_scores), np.array(ben_scores), 's3_score', 'high=suspect'


def parse_flame(log_path):
    """Parse CLIENT_FLAGS + FLAME norms → per-client norms labeled adv/ben."""
    adv_norms, ben_norms = [], []
    orders = {}   # ep -> {'order': [...], 'adv_mask': [...]}
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]', line)
            if m:
                ep = int(m.group(1))
                order = m.group(2).split(',')
                mask  = list(map(int, m.group(3).split(',')))
                orders[ep] = {'order': order, 'mask': mask}
                continue
            m = re.search(r'FLAME ep=(\d+) median_norm=[\d.]+ norms=\[([^\]]+)\]', line)
            if m:
                ep    = int(m.group(1))
                norms = list(map(float, m.group(2).split(',')))
                if ep in orders:
                    for norm, is_adv in zip(norms, orders[ep]['mask']):
                        if is_adv:
                            adv_norms.append(norm)
                        else:
                            ben_norms.append(norm)
    return np.array(adv_norms), np.array(ben_norms), 'l2_norm', 'high=suspicious (FLAME clips high)'


def parse_fltrust(log_path):
    """Parse CLIENT_FLAGS + FLTrust trust lines → per-client trust scores labeled adv/ben."""
    adv_trust, ben_trust = [], []
    orders = {}
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[CLIENT_FLAGS\] ep=(\d+) order=\[([^\]]+)\] adv_mask=\[([^\]]+)\]', line)
            if m:
                ep = int(m.group(1))
                orders[ep] = list(map(int, m.group(3).split(',')))
                continue
            m = re.search(r'FLTrust ep=(\d+)[:\s].*?trust=\[([^\]]+)\]', line)
            if m:
                ep     = int(m.group(1))
                scores = list(map(float, m.group(2).split(',')))
                if ep in orders and len(scores) == len(orders[ep]):
                    for score, is_adv in zip(scores, orders[ep]):
                        if is_adv:
                            adv_trust.append(score)
                        else:
                            ben_trust.append(score)
    return np.array(adv_trust), np.array(ben_trust), 'cosine_trust', 'low=suspect'


def compute_gate(adv, ben, stat_name, higher_is_suspect, cap=None):
    """Compute Step 1 gate statistics."""
    lines = []

    if len(adv) == 0 or len(ben) == 0:
        lines.append(f'  ERROR: insufficient data (adv={len(adv)}, ben={len(ben)})')
        return lines, float('nan'), False

    # AUROC: score = stat value, label = 1 (adv) / 0 (ben)
    # For "high=suspect" stats, AUROC > 0.5 means attackers score higher.
    # For "low=suspect" stats, flip so AUROC > 0.5 still means "defense can discriminate."
    all_scores = np.concatenate([adv, ben])
    all_labels = np.array([1]*len(adv) + [0]*len(ben))

    try:
        raw_auroc = roc_auc_score(all_labels, all_scores)
    except Exception as e:
        lines.append(f'  AUROC error: {e}')
        raw_auroc = float('nan')

    # Flip if needed so AUROC > 0.5 always means "defense can discriminate"
    auroc = raw_auroc if raw_auroc >= 0.5 else 1.0 - raw_auroc
    auroc_note = '' if raw_auroc >= 0.5 else ' (raw={:.4f}, flipped — attacker LOWER on this stat)'.format(raw_auroc)

    # KS test
    ks_stat, ks_p = ks_2samp(adv, ben)

    # Overlap: % of attacker values inside benign 5th-95th percentile
    ben_p5, ben_p95 = np.percentile(ben, 5), np.percentile(ben, 95)
    overlap_pct = 100 * np.mean((adv >= ben_p5) & (adv <= ben_p95))

    # Distribution summary
    lines.append(f'  stat: {stat_name}  ({higher_is_suspect})')
    lines.append(f'  n_adv={len(adv)}  n_ben={len(ben)}')
    lines.append(f'  benign:   mean={np.mean(ben):.4f}  std={np.std(ben):.4f}  '
                 f'p5={ben_p5:.4f}  p95={ben_p95:.4f}')
    lines.append(f'  attacker: mean={np.mean(adv):.4f}  std={np.std(adv):.4f}  '
                 f'p5={np.percentile(adv,5):.4f}  p95={np.percentile(adv,95):.4f}')
    lines.append(f'  AUROC={auroc:.4f}{auroc_note}')
    lines.append(f'  KS stat={ks_stat:.4f}  p={ks_p:.2e}')
    lines.append(f'  overlap (attacker in benign p5-p95): {overlap_pct:.1f}%')

    # Cap-specific note for FLAME
    if cap is not None and stat_name == 'l2_norm':
        pct_below_cap = 100 * np.mean(ben < cap)
        lines.append(f'  NOTE: {pct_below_cap:.1f}% of benign norms also below cap={cap:.3f}')
        lines.append(f'        FLAME clips HIGH norms only — does not act on low norms.')
        lines.append(f'        Attack norm is separable (AUROC high) but FLAME cannot catch it.')

    separable = (auroc - 0.5) > (AUROC_GATE - 0.5)
    verdict = 'SEPARABLE — gate FAILS' if separable else 'INDISTINGUISHABLE — gate PASSES'
    lines.append(f'  VERDICT: {verdict}')

    return lines, auroc, not separable


def threshold_sweep(adv, ben, stat_name, higher_is_suspect):
    """Step 2: sweep threshold, return (det_rate, fp_rate) pairs."""
    # For "high=suspect": threshold is lower bound; clients ABOVE threshold are excluded
    # For "low=suspect": threshold is upper bound; clients BELOW threshold are excluded
    all_vals = np.concatenate([adv, ben])
    thresholds = np.percentile(all_vals, np.arange(0, 101, 1))
    thresholds = np.unique(thresholds)

    pairs = []
    for t in thresholds:
        if higher_is_suspect == 'high=suspect':
            det = np.mean(adv > t)
            fp  = np.mean(ben > t)
        else:
            det = np.mean(adv < t)
            fp  = np.mean(ben < t)
        pairs.append((t, det, fp))
    return pairs


def format_sweep(pairs, stat_name, checkpoints=(0.5, 0.8, 1.0)):
    """Report honest-exclusion rate at key detection checkpoints."""
    lines = ['  Threshold sweep (det_rate → honest_fp_rate):']
    for target_det in checkpoints:
        # Find threshold nearest to achieving target_det
        candidates = [(abs(det - target_det), fp, t) for t, det, fp in pairs]
        if not candidates:
            continue
        _, fp_at, t_at = min(candidates)
        lines.append(f'    det={target_det:.0%}: honest_exclusion={fp_at:.3f}  '
                     f'(threshold={t_at:.4f})')

    # Also print a few key (det, fp) pairs
    lines.append('  Selected ROC points (det, fp):')
    step = max(1, len(pairs) // 10)
    for t, det, fp in pairs[::step]:
        lines.append(f'    t={t:.4f}  det={det:.3f}  fp={fp:.3f}')
    return lines


def main():
    global AUROC_GATE
    p = argparse.ArgumentParser()
    p.add_argument('--defense', required=True, choices=['filterfl', 'flame', 'fltrust'])
    p.add_argument('--log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cap', type=float, default=None)
    p.add_argument('--auroc-gate', type=float, default=AUROC_GATE)
    args = p.parse_args()

    AUROC_GATE = args.auroc_gate

    lines = [f'=== Step 1 gate: defense={args.defense}  log={args.log} ===']

    if args.defense == 'filterfl':
        adv, ben, stat, higher = parse_filterfl(args.log)
    elif args.defense == 'flame':
        adv, ben, stat, higher = parse_flame(args.log)
    else:
        adv, ben, stat, higher = parse_fltrust(args.log)

    gate_lines, auroc, passed = compute_gate(adv, ben, stat, higher, cap=args.cap)
    lines.extend(gate_lines)

    if passed and len(adv) > 0 and len(ben) > 0:
        lines.append('')
        lines.append('Gate PASSED → Step 2: threshold sweep')
        pairs = threshold_sweep(adv, ben, stat, higher)
        lines.extend(format_sweep(pairs, stat))
    elif not passed:
        lines.append('')
        lines.append('Gate FAILED → Step 2 skipped (defense CAN separate attacker).')
        lines.append('  Honest-exclusion tradeoff is moot: the defense already has signal.')
        if args.defense == 'flame':
            lines.append('  Key nuance: FLAME clips HIGH norms, NOT low norms.')
            lines.append('  The low-norm attack is invisible to FLAME-as-designed,')
            lines.append('  even though norm alone perfectly separates it from benign.')

    result = '\n'.join(lines)
    print(result)
    with open(args.out, 'w') as f:
        f.write(result + '\n')
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
