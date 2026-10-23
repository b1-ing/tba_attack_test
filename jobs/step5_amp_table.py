#!/usr/bin/env python3
"""Compile amplitude isolation final table.

Reads step5_wfc_*.txt (margins) and step5_amp_asr_*.txt (ASR/s3)
for all 6 amplitudes and prints the f059 table:
  amplitude x {init_margin, final_margin, below_rho?, final_ASR}
"""
import os, re, sys

BASE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5'
RHO  = 0.5

ARMS = [
    # (label, amplitude_raw, wfc_txt, asr_txt, final_round)
    ('eps4',    4/255,   'step5_wfc_eps4.txt',    'step5_amp_asr_eps4.txt',    400),
    ('eps8',    8/255,   'step5_wfc_eps8.txt',    'step5_amp_asr_eps8.txt',    400),
    ('eps16',  16/255,   'step5_wfc_eps16.txt',   'step5_amp_asr_eps16.txt',   600),
    ('checker',0.1145,   None,                    'step5_amp_asr_checker.txt', 400),
    ('eps32',  32/255,   'step5_wfc_eps32.txt',   'step5_amp_asr_eps32.txt',   600),
    ('eps64',  64/255,   'step5_wfc_eps64.txt',   'step5_amp_asr_eps64.txt',   600),
]
# checker margin from margin_decomp output (wfc sweep didn't re-run checker)
CHECKER_WFC = 'step5_wfc_checker2.txt'   # may not exist; fall back to hard-coded

def parse_wfc(path, init_rnd=50, final_rnd=None):
    """Return (init_margin, final_margin) from step5_wfc_*.txt."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        m = re.match(r'\s+(\d+)\s+([-\d.]+)\s+', line)
        if m:
            rnd, actual = int(m.group(1)), float(m.group(2))
            data[rnd] = actual
    if not data:
        return None, None
    init = data.get(init_rnd)
    final_rnd_key = final_rnd if final_rnd in data else max(data.keys())
    final = data.get(final_rnd_key)
    return init, final

def parse_asr(path, final_rnd=None):
    """Return (final_asr, final_s3, below_rho) from step5_amp_asr_*.txt."""
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        m = re.match(r'\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+|nan)\s*(.*)', line)
        if m:
            rnd  = int(m.group(1))
            asr  = float(m.group(2))
            s3   = float(m.group(4)) if m.group(4) != 'nan' else float('nan')
            note = m.group(5).strip()
            data[rnd] = (asr, s3, note)
    if not data:
        return None, None, None
    rnd_key = final_rnd if final_rnd in data else max(data.keys())
    asr, s3, note = data[rnd_key]
    below_rho = 'evades' in note if note else (s3 < RHO if s3 == s3 else None)
    return asr, s3, below_rho

# ── build table ───────────────────────────────────────────────────────────────
rows = []
for label, amp, wfc_file, asr_file, final_rnd in ARMS:
    # margins
    if label == 'checker':
        # try a dedicated checker wfc file; if not, use hard-coded known values
        init_m, final_m = parse_wfc(os.path.join(BASE, CHECKER_WFC))
        if init_m is None:
            # known values from margin_decomp.py run (probes_checker2, rounds 50/150/330/400)
            init_m, final_m = 5.129, 0.835
    else:
        wfc_path = os.path.join(BASE, wfc_file) if wfc_file else None
        init_m, final_m = parse_wfc(wfc_path, final_rnd=final_rnd) if wfc_path else (None, None)

    # ASR and s3
    asr_path = os.path.join(BASE, asr_file)
    final_asr, final_s3, below_rho = parse_asr(asr_path, final_rnd=final_rnd)

    rows.append((label, amp, init_m, final_m, final_s3, below_rho, final_asr, final_rnd))

# ── print ─────────────────────────────────────────────────────────────────────
print(f'\nAmplitude isolation: checker trigger, amplitude axis')
print(f'Setup: check12b, 100 clients, 10 ADV, poison_rate=0.5, seed=1, FilterFL (ours)')
print(f'rho=0.5 (rounds 1-500). adv_local s3 vs rho = caught/evades.')
print(f'ASR from global model (eval mode). Margin from adv_local (train mode).\n')

hdr = f'  {"Label":>12}  {"amp/255":>8}  {"init_m(r50)":>12}  {"final_m":>9}  ' \
      f'{"s3_adv_r?":>10}  {"caught?":>8}  {"ASR_global":>11}  r_final'
print(hdr)
print('  ' + '-'*90)

for label, amp, init_m, final_m, s3, below_rho, fasr, fr in rows:
    im_s  = f'{init_m:.3f}'  if init_m  is not None else '???'
    fm_s  = f'{final_m:.3f}' if final_m is not None else '???'
    s3_s  = f'{s3:.4f}'      if s3 is not None and s3 == s3 else '???'
    br_s  = 'evades' if below_rho is True else ('caught' if below_rho is False else '???')
    ar_s  = f'{fasr:.3f}'    if fasr is not None else '???'
    amp_n = amp * 255
    print(f'  {label:>12}  {amp_n:>8.2f}  {im_s:>12}  {fm_s:>9}  '
          f'{s3_s:>10}  {br_s:>8}  {ar_s:>11}  r{fr}')

print()
# crossover summary
evading = [(l, a*255, fm) for l, a, im, fm, s3, br, fa, fr in rows if br is True]
caught  = [(l, a*255, fm) for l, a, im, fm, s3, br, fa, fr in rows if br is False]
if evading and caught:
    print(f'Evasion (adv_local s3 < rho): {[l for l,_,_ in evading]}')
    print(f'Caught  (adv_local s3 > rho): {[l for l,_,_ in caught]}')
