#!/usr/bin/env python3
"""Analyse poison_grid.py output and score it against the pre-registered prediction (prediction.txt).

    python jobs/grid_analyse.py results/grid_e100 [results/grid_e1050 ...]

Needs numpy; plots (grid_<ckpt>.png in each dir) only if matplotlib is installed.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float('nan')
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


def analyse(d):
    with open(os.path.join(d, 'grid.csv')) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k]) if k != 'oracle_check' else r[k]
    rho, ck = rows[0]['rho'], int(rows[0]['ckpt_epoch'])
    cells = defaultdict(list)
    for r in rows:
        cells[(r['poison_epochs'], r['poison_rate'])].append(r)
    pes = sorted({k[0] for k in cells})
    rates = sorted({k[1] for k in cells})

    print(f'\n=== {d}: checkpoint epoch {ck}, rho = {rho}, {len(rows)} runs (all oracle-checked) ===')
    print(f'{"pe":>3} {"rate":>5} | {"p0 mean":>14} | {"frac>0.5":>14} | {"filtered":>8} | {"local ASR%":>10} | {"clean%":>6}')
    verdict = {'P1': [], 'P2': [], 'P3': []}
    for pe in pes:
        mean_p0, mean_fr, filt_rate = [], [], []
        for rt in rates:
            c = cells[(pe, rt)]
            p0 = np.array([r['att_p0_mean'] for r in c])
            fr = np.array([r['att_frac_gt05'] for r in c])
            fl = np.array([r['filtered'] for r in c])
            mean_p0.append(p0.mean()); mean_fr.append(fr.mean()); filt_rate.append(fl.mean())
            print(f'{int(pe):>3} {rt:>5.2f} | {p0.mean():6.3f}±{p0.std():5.3f} | {fr.mean():6.3f}±{fr.std():5.3f} | '
                  f'{fl.mean():8.2f} | {np.mean([r["att_local_asr"] for r in c]):10.2f} | '
                  f'{np.mean([r["att_local_clean_acc"] for r in c]):6.2f}')
        mono_p0 = bool(np.all(np.diff(mean_p0) >= 0))
        mono_fr = bool(np.all(np.diff(mean_fr) >= 0))
        rho_p0 = spearman([r['poison_rate'] for k, c in cells.items() if k[0] == pe for r in c],
                          [r['att_p0_mean'] for k, c in cells.items() if k[0] == pe for r in c])
        rho_fr = spearman([r['poison_rate'] for k, c in cells.items() if k[0] == pe for r in c],
                          [r['att_frac_gt05'] for k, c in cells.items() if k[0] == pe for r in c])
        cross = next((rt for rt, m in zip(rates, mean_p0) if m > rho), None)
        flip = next((rt for rt, m in zip(rates, filt_rate) if m >= 0.5), None)
        print(f'    pe={int(pe)}: monotone p0={mono_p0} frac={mono_fr} | spearman(rate,p0)={rho_p0:.2f} '
              f'(rate,frac)={rho_fr:.2f} | mean p0 first > rho at rate {cross} | majority-filtered from rate {flip}')
        verdict['P1'].append(mono_p0 and mono_fr)
        verdict['P3'].append(cross == flip)
        if 0.25 in rates and 0.30 in rates:
            verdict['P2'].append(mean_p0[rates.index(0.25)] < rho <= mean_p0[rates.index(0.30)])

    ben = np.mean([r['benign_p0_mean_avg'] for r in rows])
    ben_fr = np.mean([r['benign_frac_gt05_avg'] for r in rows])
    ben_max = max(r['benign_stat_max'] for r in rows)
    print(f'benign contrast: mean p0 {ben:.3f}, frac>0.5 {ben_fr:.3f}, max benign filter-stat {ben_max:.3f} (rho {rho})')
    print('prediction check: P1 monotone (all pe): ' + str(all(verdict['P1'])) +
          ' | P3 acceptance flips where mean crosses rho (all pe): ' + str(all(verdict['P3'])) +
          (' | P2 cross in (0.25, 0.30] (all pe): ' + str(all(verdict['P2'])) if verdict['P2'] and ck <= 500
           else ' | P2 n/a (late checkpoint)'))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for pe in pes:
        ax[0].plot(rates, [np.mean([r['att_p0_mean'] for r in cells[(pe, rt)]]) for rt in rates], 'o-', label=f'poison_epochs={int(pe)}')
        ax[1].plot(rates, [np.mean([r['att_frac_gt05'] for r in cells[(pe, rt)]]) for rt in rates], 'o-')
    ax[0].axhline(rho, color='k', ls='--', label=f'rho={rho}')
    ax[0].axhline(ben, color='grey', ls=':', label='benign mean')
    ax[1].axhline(ben_fr, color='grey', ls=':')
    ax[0].set(xlabel='poison rate', ylabel='pooled mean target-class prob')
    ax[1].set(xlabel='poison rate', ylabel='fraction of probes with p0 > 0.5')
    ax[0].legend(fontsize=8)
    fig.suptitle(f'checkpoint epoch {ck}')
    fig.tight_layout()
    fig.savefig(os.path.join(d, f'grid_e{ck}.png'), dpi=150)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        analyse(d)
