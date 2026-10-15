#!/usr/bin/env python3
"""Summarise FilterFL run logs (saved_models/model_*/log.txt) into per-epoch CSVs and a comparison table.

Usage:
    python jobs/log_results.py RUN_DIR_OR_LOG [RUN_DIR_OR_LOG ...] [-o OUTDIR] [--last N]

A RUN_DIR is a scratch dir made by cifar_nodef_vs_ours.sh (e.g. .../ffl2_cifar_<jobid>_ours);
the log.txt under its saved_models/ is found automatically. Labels come from the dir name
(or parent dirs for a bare log.txt). Standard library only.
"""
import argparse
import csv
import glob
import os
import re
import statistics

TEST_RE = re.compile(r'___Global_Test_Poison:(True|False)\s+Average loss: ([\d.naninf]+) Accuracy: (\d+)/(\d+) \(([\d.]+)%\)')
EPOCH_RE = re.compile(r'Epoch (\d+) completed in ([\d.]+) seconds')
DEFENCE_RE = re.compile(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)')


def find_log(path):
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, 'saved_models', 'model_*', 'log.txt')))
    if not hits:
        raise FileNotFoundError(f'no log.txt under {path}')
    return hits[-1]


def label_for(path):
    p = os.path.abspath(path)
    if os.path.isfile(p):
        p = os.path.dirname(os.path.dirname(os.path.dirname(p)))  # .../<run>/saved_models/model_x/log.txt
    return os.path.basename(p)


def parse(log_path):
    """Return per-epoch rows. Per epoch the log has: clean test, poison test, 'Epoch N completed'."""
    rows, cur = [], {}
    defence = {}
    with open(log_path, errors='replace') as f:
        for line in f:
            m = TEST_RE.search(line)
            if m:
                key = 'poison' if m.group(1) == 'True' else 'clean'
                cur[f'{key}_loss'] = float(m.group(2))
                cur[f'{key}_acc'] = float(m.group(5))
                continue
            m = DEFENCE_RE.search(line)
            if m:
                defence = dict(zip(('attacks_seen', 'filtered', 'filtered_correct'), map(int, m.groups())))
                continue
            m = EPOCH_RE.search(line)
            if m:
                if 'clean_acc' in cur and 'poison_acc' in cur:
                    rows.append({'epoch': int(m.group(1)), 'clean_acc': cur['clean_acc'], 'asr': cur['poison_acc'],
                                 'clean_loss': cur['clean_loss'], 'secs': float(m.group(2)), **defence})
                cur = {}
    return rows


def summarise(rows, last_n):
    tail = rows[-last_n:]
    best = max(rows, key=lambda r: r['clean_acc'])
    s = {
        'epochs': len(rows),
        'final_acc': rows[-1]['clean_acc'], 'final_asr': rows[-1]['asr'],
        f'mean_acc_last{len(tail)}': statistics.mean(r['clean_acc'] for r in tail),
        f'mean_asr_last{len(tail)}': statistics.mean(r['asr'] for r in tail),
        'best_acc': best['clean_acc'], 'best_acc_epoch': best['epoch'],
        'max_asr': max(r['asr'] for r in rows),
        'hours': sum(r['secs'] for r in rows) / 3600,
    }
    d = rows[-1]
    if 'filtered' in d:
        s['filtered'], s['filtered_correct'], s['attacks_seen'] = d['filtered'], d['filtered_correct'], d['attacks_seen']
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('-o', '--outdir', default='results')
    ap.add_argument('--last', type=int, default=10, help='average the final N epochs (default 10)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summaries = {}
    for run in args.runs:
        try:
            log = find_log(run)
        except FileNotFoundError as e:
            print(f'skip: {e}')
            continue
        rows = parse(log)
        if not rows:
            print(f'skip: {log} has no completed epochs yet')
            continue
        label = label_for(run)
        out = os.path.join(args.outdir, f'{label}_epochs.csv')
        fields = list(rows[-1].keys())
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, restval='')
            w.writeheader()
            w.writerows(rows)
        summaries[label] = summarise(rows, args.last)
        print(f'{label}: {len(rows)} epochs -> {out}')

    if not summaries:
        return
    keys = list(dict.fromkeys(k for s in summaries.values() for k in s))
    with open(os.path.join(args.outdir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run'] + keys)
        for label, s in summaries.items():
            w.writerow([label] + [f'{s[k]:.2f}' if isinstance(s.get(k), float) else s.get(k, '') for k in keys])

    print()
    width = max(len(k) for k in keys)
    labels = list(summaries)
    print(' ' * width, *[f'{l[-24:]:>24}' for l in labels])
    for k in keys:
        cells = [(f'{summaries[l][k]:.2f}' if isinstance(summaries[l].get(k), float) else str(summaries[l].get(k, '-'))) for l in labels]
        print(f'{k:<{width}}', *[f'{c:>24}' for c in cells])
    print(f"\nWrote {os.path.join(args.outdir, 'summary.csv')}")


if __name__ == '__main__':
    main()
