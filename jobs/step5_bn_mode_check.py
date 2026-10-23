"""BN-mode comparison: train vs eval s3 on the FilterFL probe.

For each saved round, evaluates the probe in both TRAIN mode (as algorithm.py Stage 3 does)
and EVAL mode (running stats) on two model types:
  global_model  d2 from model_ep{N}.pt  — proxy for benign client (no poisoning)
  adv_local     adv_local_ep{N}.pt      — adversarial client's local model

Hypothesis (BN-noise cause of benign false positives):
  global train_s3 > rho  AND  global eval_s3 < rho  →  BN batch-stat noise confirmed
  adv    train_s3 > rho  AND  adv    eval_s3 > rho  →  adversarial detection is real, not noise

Usage:
  python3 step5_bn_mode_check.py <probe_dir> <filterfl_dir>

  probe_dir:    ~/ffl_step5/probes_checker2  (has model_ep{N}.pt, adv_local_ep{N}.pt,
                                              pixel_probe_ep{N}.pt)
  filterfl_dir: ~/tba_attack_test/FilterFL
"""

import sys, os, glob, re
import torch
import torch.nn.functional as F

probe_dir = sys.argv[1]
ffl_dir   = sys.argv[2]

sys.path.insert(0, ffl_dir)
import trigger_utils as tu
from models.ResNet8 import ResNet8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TARGET = 0

RHO_EARLY = 0.50   # rounds 1-500
RHO_LATE  = 0.40   # rounds 501+

print(f'probe_dir: {probe_dir}')
print(f'trigger_utils v{tu.VERSION}  device: {device}')
print(f'rho: {RHO_EARLY} (rounds ≤500), {RHO_LATE} (rounds >500)')
print()


def s3_score(state_dict, probe, eval_mode):
    """Load state, set mode, return s3 = softmax(model(probe)).mean(0)[TARGET]."""
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    if eval_mode:
        m.eval()
    # else: stays in train mode (default after load_state_dict) — matches algorithm.py line 187
    with torch.no_grad():
        out = torch.softmax(m(probe), dim=1).mean(0)
        s3 = out[TARGET].item()
        argmax = out.argmax().item()
    return s3, argmax


# Discover available rounds from probe files
probe_files = sorted(glob.glob(os.path.join(probe_dir, 'pixel_probe_ep*.pt')))
rounds = []
for pf in probe_files:
    ep = int(re.search(r'ep(\d+)', os.path.basename(pf)).group(1))
    rounds.append(ep)

print(f'Available probe rounds: {rounds}')
print()

header = (f'{"Rnd":>5}  {"Model":>12}  '
          f'{"s3_train":>9}  {"arg_train":>10}  '
          f'{"s3_eval":>9}  {"arg_eval":>10}  '
          f'{"delta":>8}  {"rho":>5}  note')
print(header)
print('-' * 95)

for rnd in rounds:
    rho = RHO_EARLY if rnd <= 500 else RHO_LATE
    probe_path = os.path.join(probe_dir, f'pixel_probe_ep{rnd:04d}.pt')
    probe = torch.load(probe_path, map_location='cpu', weights_only=True).to(device)

    # Global model (d2 = new aggregated global after this round — benign proxy)
    model_path = os.path.join(probe_dir, f'model_ep{rnd:04d}.pt')
    if os.path.exists(model_path):
        m_data = torch.load(model_path, map_location='cpu', weights_only=False)
        d2 = m_data['d2']   # aggregated global model
        s3_tr, arg_tr = s3_score(d2, probe, eval_mode=False)
        s3_ev, arg_ev = s3_score(d2, probe, eval_mode=True)
        delta = s3_tr - s3_ev
        cross_tr = '⚠ CROSS' if s3_tr > rho and arg_tr == TARGET else ''
        cross_ev = '✓ ok'    if s3_ev <= rho or arg_ev != TARGET else '⚠ CROSS'
        print(f'  {rnd:4d}  {"global(d2)":>12}  '
              f'{s3_tr:>9.4f}  {arg_tr:>10}  '
              f'{s3_ev:>9.4f}  {arg_ev:>10}  '
              f'{delta:>+8.4f}  {rho:>5.2f}  train:{cross_tr} eval:{cross_ev}',
              flush=True)

    # Adversarial local model
    adv_path = os.path.join(probe_dir, f'adv_local_ep{rnd:04d}.pt')
    if os.path.exists(adv_path):
        adv_state = torch.load(adv_path, map_location='cpu', weights_only=False)
        s3_tr, arg_tr = s3_score(adv_state, probe, eval_mode=False)
        s3_ev, arg_ev = s3_score(adv_state, probe, eval_mode=True)
        delta = s3_tr - s3_ev
        cross_tr = '⚠ CROSS' if s3_tr > rho and arg_tr == TARGET else ''
        cross_ev = '✓ ok'    if s3_ev <= rho or arg_ev != TARGET else '⚠ CROSS'
        print(f'  {rnd:4d}  {"adv_local":>12}  '
              f'{s3_tr:>9.4f}  {arg_tr:>10}  '
              f'{s3_ev:>9.4f}  {arg_ev:>10}  '
              f'{delta:>+8.4f}  {rho:>5.2f}  train:{cross_tr} eval:{cross_ev}',
              flush=True)
    else:
        print(f'  {rnd:4d}  {"adv_local":>12}  (not yet saved)', flush=True)

    print(flush=True)

print()
print('Interpretation:')
print('  global train_s3 > rho AND global eval_s3 < rho  →  BN batch-stat noise confirmed')
print('  adv    train_s3 > rho AND adv    eval_s3 > rho  →  adversarial detection is real')
print('  delta = s3_train - s3_eval; large positive delta = train-mode BN inflates s3')
