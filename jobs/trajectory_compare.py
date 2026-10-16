#!/usr/bin/env python3
"""Attacker p0, benign p0 and AUC(p0: attacker vs benign) in 100-round windows.

Runs compared:
  blind-spot (filterfl_test, rate 0.25, seed 0, 1000 rds): probs (N,32,10) -> p0 = probs[:,:,0].mean(1)
  ref rate 0.25 seed 1 (reference FilterFL, 2000 rds): probs (N,10)
  ref rate 0.3 seed 1 (reference FilterFL, 2000 rds): probs (N,10)
  ref rate 0.3 seed 2 (reference FilterFL, 2000 rds): probs (N,10)

    python jobs/trajectory_compare.py OUT_DIR      # writes trajectory_compare.png and .csv
"""
import csv
import os
import sys

import numpy as np

HOME = os.path.expanduser('~')
REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'probe')
BLIND = f'{HOME}/filterfl_test/filterfl_test/results/hpc_s10_cifar_mean_a_s0_probe.npz'
RUNS = [  # label, source, kind
    ('blind-spot r0.25 s0 (filterfl_test, 1000 rds)', BLIND, 'blind'),
    ('ref r0.25 s1 (reference code)', os.path.join(REPO, 's1_ours_r0.25', 'probes.npz'), 'ref'),
    ('ref r0.3 s1 (reference code)', os.path.join(REPO, 's1_ours', 'probes.npz'), 'ref'),
    ('ref r0.3 s2 (reference code)', os.path.join(REPO, 's2_ours', 'probes.npz'), 'ref'),
]
W = 100


def load(path, kind):
    d = np.load(path)
    if kind == 'blind':
        return d['epoch'], d['is_mal'], d['probs'][:, :, 0].mean(1)
    return d['epoch'], d['is_adv'], d['probs'][:, 0]


def auc(a, b):  # P(attacker p0 > benign p0), ties half
    r = np.argsort(np.argsort(np.concatenate([a, b]))).astype(float)
    return (r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b))


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows, series = [], {}
for label, path, kind in RUNS:
    e, adv, p0 = load(path, kind)
    s = {'x': [], 'att': [], 'ben': [], 'auc': []}
    for w in range(1, int(e.max()) + 1, W):
        m = (e >= w) & (e < w + W)
        a, b = p0[m & adv], p0[m & ~adv]
        if len(a) < 5:
            continue
        s['x'].append(w + W / 2 - 0.5); s['att'].append(np.median(a)); s['ben'].append(np.median(b)); s['auc'].append(auc(a, b))
        rows.append([label, w, w + W - 1, len(a), len(b), s['att'][-1], s['ben'][-1], s['auc'][-1]])
    series[label] = s
with open(os.path.join(out, 'trajectory_compare.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['run', 'win_start', 'win_end', 'n_attacker_rows', 'n_benign_rows', 'attacker_p0_median', 'benign_p0_median', 'auc'])
    wr.writerows([[r[0], r[1], r[2], r[3], r[4], f'{r[5]:.4f}', f'{r[6]:.4f}', f'{r[7]:.4f}'] for r in rows])

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

COL = ['#eb6834', '#d6862a', '#2a78d6', '#1baf7a']   # warm=blind-spot, amber=ref r0.25, blue/green=r0.3
STY = ['--', '-', '-', '--']
INK, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e0'
fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True, facecolor=SURF)
for ax, key, title, ylab in zip(axs, ['att', 'ben', 'auc'],
                                ['Attacker p0 (median per 100-round window)', 'Benign p0 (median per 100-round window)',
                                 'AUC of p0, attacker vs benign (per 100-round window)'], ['p0', 'p0', 'AUC']):
    ax.set_facecolor(SURF)
    for (label, _, _), c, ls in zip(RUNS, COL, STY):
        s = series[label]
        ax.plot(s['x'], s[key], ls, color=c, lw=2, label=label, marker='o', ms=4, mfc=SURF, mew=1.5)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_ylabel(ylab, color=MUTED)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED)
axs[0].set_ylim(0, 0.9); axs[1].set_ylim(0, 0.9); axs[2].set_ylim(0.5, 1.02)
axs[2].axhline(0.5, color=MUTED, lw=0.8, ls=':')
axs[2].set_xlabel('round (window centre)', color=MUTED); axs[2].set_xlim(0, 2000)
axs[0].legend(frameon=False, loc='upper right', fontsize=9, labelcolor=INK)
fig.text(0.01, 0.005,
         'Dashed: filterfl_test codebase (seed 0). Solid: reference FilterFL (seed 1). '
         'All runs: CIFAR-10, alpha=0.5. Threshold rho not drawn.',
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig(os.path.join(out, 'trajectory_compare.png'), dpi=150, facecolor=SURF)
print('wrote', out)
