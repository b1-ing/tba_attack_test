"""F-adv scatter: FEAT_ALIGN vs s3 for checker and patch adversarial clients.

Reads FEAT_ALIGN lines from both arms' logs and outputs:
  - Pearson r(fa, s3) per arm
  - Combined table for scatter
  - F-ben benign crossing rate per arm (from FEAT_ALIGN_BEN lines)

Usage:
  python3 step5_fa_scatter.py \\
      --checker_log  ~/ffl_step5/step5_probespectrum_checker.log \\
      --patch_log    ~/ffl_step5/step5_probespectrum_patch2.log \\
      [--rho_early 0.50] [--rho_late 0.40]

  Or with pre-extracted files:
      --checker_adv  ~/ffl_step5/step5_checker_fa_adv.txt   (optional, uses log if absent)
      --checker_ben  ~/ffl_step5/step5_checker_fa_ben.txt
      --patch_adv    ~/ffl_step5/step5_patch2_fa_adv.txt
      --patch_ben    ~/ffl_step5/step5_patch2_fa_ben.txt
"""

import argparse, re, sys
from collections import defaultdict

def parse_fa_lines(path, is_ben=False):
    """Parse FEAT_ALIGN or FEAT_ALIGN_BEN lines. Returns list of (ep, client, fa, s3)."""
    pattern = re.compile(
        r'FEAT_ALIGN(?:_BEN)? ep=(\d+) client=(\S+) fa=(-?[\d.]+) s3=([\d.]+)'
    )
    records = []
    try:
        with open(path) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    ep, client, fa, s3 = m.groups()
                    records.append((int(ep), client, float(fa), float(s3)))
    except FileNotFoundError:
        print(f'  (file not found: {path})', file=sys.stderr)
    return records


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den > 0 else float('nan')


def ben_crossing_rate(records, rho_early, rho_late):
    total = len(records)
    crossers = [(ep, c, fa, s3) for ep, c, fa, s3 in records
                if s3 > (rho_early if ep <= 500 else rho_late)]
    n_cross = len(crossers)
    pct = 100.0 * n_cross / total if total else 0.0
    fa_neg = sum(1 for _, _, fa, _ in crossers if fa < 0)
    pct_neg = 100.0 * fa_neg / n_cross if n_cross else 0.0
    top = sorted(crossers, key=lambda x: -x[3])[:3]
    return total, n_cross, pct, fa_neg, pct_neg, top


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checker_log',  default=None)
    p.add_argument('--patch_log',    default=None)
    p.add_argument('--checker_adv',  default=None)
    p.add_argument('--checker_ben',  default=None)
    p.add_argument('--patch_adv',    default=None)
    p.add_argument('--patch_ben',    default=None)
    p.add_argument('--rho_early', type=float, default=0.50)
    p.add_argument('--rho_late',  type=float, default=0.40)
    args = p.parse_args()

    arms = {}
    for arm, log_path, adv_path, ben_path in [
        ('checker', args.checker_log, args.checker_adv, args.checker_ben),
        ('patch',   args.patch_log,   args.patch_adv,   args.patch_ben),
    ]:
        # Prefer pre-extracted files; fall back to full log
        adv_src = adv_path or log_path
        ben_src = ben_path or log_path
        if adv_src is None:
            print(f'{arm}: no source provided, skipping.')
            continue

        print(f'\n=== {arm.upper()} arm ===')

        # ADV records
        adv = parse_fa_lines(adv_src, is_ben=False)
        # If reading from full log, filter to ADV lines only (no _BEN)
        if adv_path is None and log_path:
            adv = [(ep, c, fa, s3) for ep, c, fa, s3 in adv
                   if True]  # parse_fa_lines already returns both; separate below
            # Re-parse properly
            adv = []
            ben_from_log = []
            adv_pat = re.compile(r'FEAT_ALIGN ep=(\d+) client=(\S+) fa=(-?[\d.]+) s3=([\d.]+)')
            ben_pat = re.compile(r'FEAT_ALIGN_BEN ep=(\d+) client=(\S+) fa=(-?[\d.]+) s3=([\d.]+)')
            try:
                with open(log_path) as f:
                    for line in f:
                        ma = adv_pat.search(line)
                        mb = ben_pat.search(line)
                        if ma:
                            ep, client, fa, s3 = ma.groups()
                            adv.append((int(ep), client, float(fa), float(s3)))
                        elif mb:
                            ep, client, fa, s3 = mb.groups()
                            ben_from_log.append((int(ep), client, float(fa), float(s3)))
            except FileNotFoundError:
                print(f'  log not found: {log_path}')
                continue
            ben = ben_from_log
        else:
            ben = parse_fa_lines(ben_src, is_ben=True) if ben_src else []
            # filter: adv file has only ADV lines, ben file has only BEN lines
            # parse_fa_lines works for both; no further filtering needed

        if not adv:
            print(f'  No ADV FEAT_ALIGN records found.')
        else:
            fa_vals = [r[2] for r in adv]
            s3_vals = [r[3] for r in adv]
            r = pearson_r(fa_vals, s3_vals)
            print(f'  ADV records: {len(adv)}')
            print(f'  fa range: [{min(fa_vals):.4f}, {max(fa_vals):.4f}]')
            print(f'  s3 range: [{min(s3_vals):.4f}, {max(s3_vals):.4f}]')
            print(f'  Pearson r(fa, s3): {r:+.4f}')
            # Round-by-round mean
            by_ep = defaultdict(list)
            for ep, _, fa, s3 in adv:
                by_ep[ep].append((fa, s3))
            print(f'  Round means (sampled):')
            for ep in sorted(by_ep)[:10]:
                fas = [x[0] for x in by_ep[ep]]
                s3s = [x[1] for x in by_ep[ep]]
                print(f'    ep={ep:4d}  n={len(fas)}  fa={sum(fas)/len(fas):.4f}  s3={sum(s3s)/len(s3s):.4f}')
            arms[arm] = {'adv': adv, 'r': r}

        if not ben:
            print(f'  No BEN FEAT_ALIGN_BEN records found.')
        else:
            total, n_cross, pct, fa_neg, pct_neg, top = ben_crossing_rate(
                ben, args.rho_early, args.rho_late)
            print(f'  BEN records: {total}')
            print(f'  Crossers (s3>rho): {n_cross}/{total} = {pct:.1f}%')
            print(f'  Of crossers with fa<0: {fa_neg}/{n_cross} = {pct_neg:.1f}%')
            print(f'  Top crossers (by s3):')
            for ep, c, fa, s3 in top:
                rho = args.rho_early if ep <= 500 else args.rho_late
                print(f'    ep={ep:4d} client={c} s3={s3:.4f} fa={fa:.4f} rho={rho}')
            arms[arm] = arms.get(arm, {})
            arms[arm]['ben_total'] = total
            arms[arm]['ben_cross'] = n_cross
            arms[arm]['ben_pct'] = pct

    # Combined comparison
    if len(arms) == 2 and 'checker' in arms and 'patch' in arms:
        print('\n=== COMPARISON ===')
        for arm in ['checker', 'patch']:
            a = arms[arm]
            n_adv = len(a.get('adv', []))
            r = a.get('r', float('nan'))
            pct = a.get('ben_pct', float('nan'))
            print(f'  {arm:8s}  ADV n={n_adv:5d}  r(fa,s3)={r:+.4f}  '
                  f'BEN FP={pct:.1f}%')
        print()
        print('  Prediction: r>+0.7 for both arms -> F-adv generalises across triggers')
        print('  Prediction: BEN FP ~15% for both -> BN noise is trigger-independent')


if __name__ == '__main__':
    main()
