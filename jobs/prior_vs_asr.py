#!/usr/bin/env python3
"""Checks 1 & 2: does the Stage-3 detector track prior (label-distribution skew)
rather than backdoor effectiveness?

Check 1 — across poison rates
  For adversarial clients at rate 0.25, 0.30, 0.50: compare
    - Stage-3 score  (mean probs[0] on GAN probe batch — what the detector sees)
    - ASR            (per-epoch backdoor success rate from probe logs)
  Prediction: Stage-3 score increases monotonically with rate; ASR saturates.

Check 2 — false-positive benign clients
  For benign clients that were nonetheless flagged (probs[0] > threshold):
    - compare their Stage-3 score distribution to flagged adversarial clients.
  Prediction: the distributions overlap, demonstrating the detector cannot
  distinguish a skewed benign from a malicious client.

Usage:
    python jobs/prior_vs_asr.py \\
        --r025 ~/ffl_probe/s1_ours_r0.25/probes.npz \\
        --r030 ~/ffl_probe/s1_ours/probes.npz \\
        --r050 ~/ffl_probe/s1_ours_r0.5/probes.npz \\
        --out  results/analysis/prior_vs_asr
"""
import argparse
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def stage3_stats(d, last_n_epochs=None):
    """Per-client Stage-3 stats over (optionally) the last N epochs."""
    epoch = d['epoch']
    if last_n_epochs is not None:
        max_ep = epoch.max()
        mask = epoch > (max_ep - last_n_epochs)
        p0     = d['probs'][mask, 0]
        is_adv = d['is_adv'][mask]
        std    = d['standard'][mask]
    else:
        p0     = d['probs'][:, 0]
        is_adv = d['is_adv']
        std    = d['standard']
    return p0, is_adv, std


def check1(runs, out_dir):
    """Check 1: Stage-3 score vs ASR across poison rates."""
    print("\n=== CHECK 1: Stage-3 score vs ASR across poison rates ===")
    print(f"{'rate':>6}  {'adv_p0_mean':>12}  {'adv_p0_med':>11}  "
          f"{'flag_rate_adv':>14}  {'final_ASR%':>11}  {'flag_rate_ben':>14}")

    rates, adv_p0_means, asrs, flag_rates_adv, flag_rates_ben = [], [], [], [], []

    for rate, d in sorted(runs.items()):
        p0, is_adv, std = stage3_stats(d, last_n_epochs=500)
        adv_p0   = p0[is_adv]
        ben_p0   = p0[~is_adv]
        flag_adv = np.mean(adv_p0 > std[is_adv])
        flag_ben = np.mean(ben_p0 > std[~is_adv])
        asr      = float(d['asr'][-1]) if len(d['asr']) else float('nan')  # already %

        print(f"{rate:>6.2f}  {adv_p0.mean():>12.4f}  {np.median(adv_p0):>11.4f}  "
              f"{flag_adv:>14.3f}  {asr:>11.1f}  {flag_ben:>14.3f}")

        rates.append(rate); adv_p0_means.append(adv_p0.mean())
        asrs.append(asr); flag_rates_adv.append(flag_adv); flag_rates_ben.append(flag_ben)

    # Spearman correlation between rate and each metric
    def spearman(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    print(f"\nSpearman(rate, adv_p0_mean) = {spearman(rates, adv_p0_means):+.3f}  "
          f"[prediction: strongly positive]")
    print(f"Spearman(rate, final_ASR)   = {spearman(rates, asrs):+.3f}  "
          f"[prediction: weaker / saturates]")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))

        axes[0].plot(rates, adv_p0_means, 'o-', color='tab:red')
        axes[0].set(xlabel='poison rate', ylabel='mean Stage-3 score (probs[0])',
                    title='Stage-3 score vs poison rate')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(rates, asrs, 's-', color='tab:blue')
        axes[1].set(xlabel='poison rate', ylabel='final ASR (%)',
                    title='ASR vs poison rate')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(rates, flag_rates_adv, 'o-', color='tab:red', label='adversarial')
        axes[2].plot(rates, flag_rates_ben, 's--', color='tab:green', label='benign')
        axes[2].set(xlabel='poison rate', ylabel='fraction flagged by detector',
                    title='Detection rate vs poison rate')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        out = os.path.join(out_dir, 'check1_score_vs_asr.png')
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)


def check2(runs, out_dir):
    """Check 2: flagged benign vs flagged adversarial Stage-3 distributions."""
    print("\n=== CHECK 2: Flagged benign vs flagged adversarial distributions ===")

    # Use the middle poison rate for the main comparison
    rate_key = sorted(runs.keys())[len(runs) // 2]
    d = runs[rate_key]
    print(f"(Using poison_rate={rate_key:.2f} run)")

    p0, is_adv, std = stage3_stats(d)
    flagged   = p0 > std
    adv_flagged  = p0[ is_adv &  flagged]
    ben_flagged  = p0[~is_adv &  flagged]
    adv_unflagged = p0[ is_adv & ~flagged]

    print(f"\nFlagged adversarial clients : n={len(adv_flagged)}  "
          f"mean={adv_flagged.mean():.4f}  median={np.median(adv_flagged):.4f}")
    print(f"Flagged benign clients      : n={len(ben_flagged)}  "
          f"mean={ben_flagged.mean():.4f}  median={np.median(ben_flagged):.4f}")
    print(f"Unflagged adversarial       : n={len(adv_unflagged)}  "
          f"mean={adv_unflagged.mean():.4f}")

    # Is the benign flagged distribution distinguishable from adversarial flagged?
    if len(ben_flagged) > 0 and len(adv_flagged) > 0:
        # Mann-Whitney U without scipy
        def mannwhitney_u(a, b):
            a, b = np.asarray(a), np.asarray(b)
            na, nb = len(a), len(b)
            u = sum(np.sum(ai > b) + 0.5 * np.sum(ai == b) for ai in a)
            mu = na * nb / 2
            su = np.sqrt(na * nb * (na + nb + 1) / 12)
            z = (u - mu) / su
            # two-sided p from normal approximation
            pval = 2 * (1 - 0.5 * (1 + float(np.sign(z)) * float(
                1 - np.exp(-0.717 * abs(z) - 0.416 * z * z))))
            return u, pval
        u_stat, pval = mannwhitney_u(adv_flagged, ben_flagged)
        print(f"\nMann-Whitney U={u_stat:.0f}, p≈{pval:.4f} (flagged adv vs flagged ben)  "
              f"[indistinguishable if p >> 0.05]")
    else:
        print("(Not enough flagged benign samples for statistical test)")

    # Show false positive rate vs true positive rate at each threshold
    print("\nFalse-positive vs True-positive rate sweep:")
    print(f"{'threshold':>10}  {'TPR':>6}  {'FPR':>6}  {'flagged_adv':>12}  {'flagged_ben':>12}")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        tp = np.mean(p0[is_adv]  > thresh)
        fp = np.mean(p0[~is_adv] > thresh)
        na = np.sum(is_adv  & (p0 > thresh))
        nb = np.sum(~is_adv & (p0 > thresh))
        print(f"{thresh:>10.2f}  {tp:>6.3f}  {fp:>6.3f}  {na:>12}  {nb:>12}")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        bins = np.linspace(0, 1, 41)
        if len(adv_flagged):
            axes[0].hist(adv_flagged, bins=bins, alpha=0.6, color='tab:red',
                         density=True, label=f'flagged adversarial (n={len(adv_flagged)})')
        if len(ben_flagged):
            axes[0].hist(ben_flagged, bins=bins, alpha=0.6, color='tab:green',
                         density=True, label=f'flagged benign (n={len(ben_flagged)})')
        axes[0].set(xlabel='Stage-3 score (probs[0])',
                    title=f'Flagged distributions (rate={rate_key:.2f})')
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        # Full distributions
        axes[1].hist(p0[is_adv],  bins=bins, alpha=0.5, color='tab:red',
                     density=True, label=f'adversarial (n={is_adv.sum()})')
        axes[1].hist(p0[~is_adv], bins=bins, alpha=0.5, color='tab:green',
                     density=True, label=f'benign (n={(~is_adv).sum()})')
        axes[1].set(xlabel='Stage-3 score (probs[0])',
                    title=f'All scores (rate={rate_key:.2f})')
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        out = os.path.join(out_dir, 'check2_flag_distributions.png')
        fig.savefig(out, dpi=150)
        print(f"Saved: {out}")
        plt.close(fig)


def check2_cross_rate(runs, out_dir):
    """Supplementary: benign false-positive rate vs poison rate."""
    print("\n=== SUPPLEMENTARY: Benign false-positive rate across poison rates ===")
    print(f"{'rate':>6}  {'FPR (all epochs)':>18}  {'FPR (last 500ep)':>18}  {'n_ben_flagged':>14}")
    for rate, d in sorted(runs.items()):
        for label, last_n in [('all', None), ('last500', 500)]:
            p0, is_adv, std = stage3_stats(d, last_n)
            fpr = np.mean(p0[~is_adv] > std[~is_adv])
            n   = np.sum(~is_adv & (p0 > std))
            if label == 'all':
                print(f"{rate:>6.2f}  {fpr:>18.4f}", end='')
            else:
                print(f"  {fpr:>18.4f}  {n:>14}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r025', default=None)
    ap.add_argument('--r030', default=None)
    ap.add_argument('--r050', default=None)
    ap.add_argument('--out', default='results/analysis/prior_vs_asr')
    args = ap.parse_args()

    runs = {}
    defaults = {
        0.25: os.path.expanduser('~/ffl_probe/s1_ours_r0.25/probes.npz'),
        0.30: os.path.expanduser('~/ffl_probe/s1_ours/probes.npz'),
        0.50: os.path.expanduser('~/ffl_probe/s1_ours_r0.5/probes.npz'),
    }
    overrides = {0.25: args.r025, 0.30: args.r030, 0.50: args.r050}
    for rate, path in defaults.items():
        p = overrides[rate] or path
        if p and os.path.exists(p):
            runs[rate] = load(p)
            print(f"Loaded rate={rate:.2f}: {p}  (poison_rate in file: {runs[rate]['poison_rate']})")
        else:
            print(f"Skipping rate={rate:.2f}: not found at {p}")

    if not runs:
        print("No probe files found — pass --r025/--r030/--r050 explicitly.")
        return

    os.makedirs(args.out, exist_ok=True)

    if len(runs) >= 2:
        check1(runs, args.out)
        check2_cross_rate(runs, args.out)
    check2(runs, args.out)

    print(f"\nAll outputs written to: {args.out}")


if __name__ == '__main__':
    main()
