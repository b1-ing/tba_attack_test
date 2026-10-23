#!/usr/bin/env python3
"""Detection vs prevention analysis.

For each round, parse from the FilterFL training log:
  - n_adv_selected: adversarial clients selected this round (count_all_attack)
  - n_adv_detected: adversarial clients detected/excluded (count_correct_defense)
  - n_missed: n_adv_selected - n_adv_detected (slipped through)
  - asr: Global_Test_Poison:True accuracy (%)

Question: does ASR rise in rounds where ALL adversarial clients were excluded
(i.e., n_missed == 0)?  That would be the strong indictment: exclusion itself fails.
"""
import sys, re, os
import numpy as np

def parse_log(path):
    """Return list of dicts, one per round."""
    rounds = []
    cur = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.search(r'count_all_attack:\s*(\d+),\s*count_all_defense:\s*(\d+),\s*count_correct_defense:\s*(\d+)', line)
            if m:
                if cur:
                    rounds.append(cur)
                cur = {
                    'n_adv': int(m.group(1)),
                    'n_def': int(m.group(2)),
                    'n_correct': int(m.group(3)),
                }
                continue
            m2 = re.search(r'Global_Test_Poison:True.*?Accuracy:\s*(\d+)/(\d+)', line)
            if m2 and cur:
                cur['asr'] = int(m2.group(1)) / int(m2.group(2))
                cur['rnd'] = len(rounds) + 1  # approximate round number
    if cur and 'asr' in cur:
        rounds.append(cur)
    return rounds

def analyze(label, path):
    if not os.path.exists(path):
        print(f'{label}: log not found at {path}'); return

    rounds = parse_log(path)
    if not rounds:
        print(f'{label}: no parseable rounds'); return

    print(f'\n{"="*70}')
    print(f'{label}  ({len(rounds)} rounds parsed)')
    print(f'{"="*70}')

    # Summary stats
    all_missed = [r['n_adv'] - r['n_correct'] for r in rounds]
    all_asr    = [r['asr'] for r in rounds]

    # Rounds where 0 adversarial clients missed (all excluded)
    full_excl  = [r for r in rounds if r['n_adv'] - r['n_correct'] == 0 and r['n_adv'] > 0]
    partial    = [r for r in rounds if r['n_adv'] - r['n_correct'] > 0]
    zero_adv   = [r for r in rounds if r['n_adv'] == 0]

    print(f'\nRound classification:')
    print(f'  Rounds with ≥1 adv selected, ALL excluded (n_missed=0): {len(full_excl)}')
    print(f'  Rounds with ≥1 adv selected, SOME missed  (n_missed>0): {len(partial)}')
    print(f'  Rounds with 0 adv selected this round:                   {len(zero_adv)}')

    if full_excl:
        asr_fe = [r['asr'] for r in full_excl]
        print(f'\n  ASR in fully-excluded rounds (n_missed=0):')
        print(f'    mean={np.mean(asr_fe):.3f}  min={np.min(asr_fe):.3f}  max={np.max(asr_fe):.3f}')
        # Trend: first half vs second half
        mid = len(full_excl) // 2
        if mid > 0:
            print(f'    early half mean={np.mean(asr_fe[:mid]):.3f}  late half mean={np.mean(asr_fe[mid:]):.3f}')

    if partial:
        asr_p = [r['asr'] for r in partial]
        print(f'\n  ASR in partial-escape rounds (n_missed>0):')
        print(f'    mean={np.mean(asr_p):.3f}  min={np.min(asr_p):.3f}  max={np.max(asr_p):.3f}')

    # Overall ASR trajectory (every 50 rounds)
    print(f'\n  ASR trajectory (every 50 rounds):')
    print(f'  {"Rnd":>5}  {"n_adv":>6}  {"n_miss":>7}  {"ASR":>6}  note')
    print('  ' + '-'*45)
    step = max(1, len(rounds) // 12)
    shown = set()
    for i, r in enumerate(rounds):
        if i % step == 0 or i == len(rounds)-1:
            if i not in shown:
                shown.add(i)
                miss = r['n_adv'] - r['n_correct']
                note = 'all excluded' if miss == 0 and r['n_adv'] > 0 else \
                       f'{miss} missed' if miss > 0 else 'no adv'
                print(f'  {r["rnd"]:>5}  {r["n_adv"]:>6}  {miss:>7}  {r["asr"]:>6.3f}  {note}')

    # Key test: does ASR increase even in fully-excluded rounds?
    if len(full_excl) > 5:
        fe_rsorted = sorted(full_excl, key=lambda r: r['rnd'])
        # Compare ASR at start vs end of fully-excluded stretches
        q1 = np.percentile([r['asr'] for r in fe_rsorted[:len(fe_rsorted)//3]], 50)
        q3 = np.percentile([r['asr'] for r in fe_rsorted[2*len(fe_rsorted)//3:]], 50)
        print(f'\n  Among fully-excluded rounds: early median ASR={q1:.3f}  late median ASR={q3:.3f}')
        if q3 > q1 + 0.02:
            print(f'  -> ASR RISES in fully-excluded rounds: exclusion does NOT prevent implantation (measured)')
        elif q3 < q1 - 0.02:
            print(f'  -> ASR FALLS in fully-excluded rounds: exclusion helps (measured)')
        else:
            print(f'  -> ASR FLAT in fully-excluded rounds (measured)')

BASE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5'
ARMS = [
    ('eps4  (4/255)',    f'{BASE}/step5_wfc_eps4.log'),
    ('eps8  (8/255)',    f'{BASE}/step5_wfc_eps8.log'),
    ('checker (29/255)', f'{BASE}/step5_probespectrum_chk2.log'),
    ('eps16 (16/255)',   f'{BASE}/step5_amp_iso_eps16.log'),
    ('eps32 (32/255)',   f'{BASE}/step5_amp_iso_eps32.log'),
    ('eps64 (64/255)',   f'{BASE}/step5_amp_iso_eps64.log'),
]

for label, path in ARMS:
    analyze(label, path)

print('\nDone.')
