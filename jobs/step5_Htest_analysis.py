"""Experiment H analysis: is S3 decay model-driven (probe held fixed)?

For each round in [50, 150, 250, 330, 400], evaluates two FIXED probes on three
global-model trajectories trained for 400 rounds of FedAvg:
  checker   — backdoored (checker trigger, undefended)
  clean     — no poisoning (benign control)
  patch     — backdoored (patch trigger, undefended)

Fixed probes (built once, never reoptimised):
  probe_A: checker trigger on gray background, ImageNet-normalised
  probe_B: pixel-space probe saved at ep=50 from the probe-spectrum run (normalised)

Metrics per (round, trajectory, probe):
  s3        = softmax(model(probe)).mean(dim=0)[TARGET]   raw target-class confidence
  gap       = s3_checker_back - s3_clean                  detector's usable signal
  feat_align= cosine(mean_penultimate(model, probe), model.linear1.weight[TARGET])

Read-out:
  gap collapses across rounds (probe fixed) → H confirmed: decay is model-driven
  gap flat                                  → H refuted: decay was probe-driven
  patch gap holds while checker gap collapses → checker-specific, control valid

Usage:
  python3 step5_Htest_analysis.py <Htest_base_dir> <filterfl_dir> <pixel50_probe_path>

  Htest_base_dir: parent of checker/, clean/, patch/ subdirs from step5_Htest_train.sh
  filterfl_dir:   path to FilterFL/ (for trigger_utils and ResNet8)
  pixel50_probe_path: pixel_probe_ep0050.pt from the probe-spectrum run
"""

import sys, os
import numpy as np
import torch
import torch.nn.functional as F

base_dir       = sys.argv[1]
ffl_dir        = sys.argv[2]
pixel50_path   = sys.argv[3]

sys.path.insert(0, ffl_dir)
import trigger_utils as tu
from models.ResNet8 import ResNet8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TARGET = 0
ROUNDS = [50, 150, 250, 330, 400]

MEAN_T = torch.tensor(tu.MEAN, device=device).view(1, 3, 1, 1)
STD_T  = torch.tensor(tu.STD,  device=device).view(1, 3, 1, 1)


def _penultimate(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out); out = model.layer2(out); out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def eval_probe(state, probe_norm):
    """Return (s3, feat_align) for a state dict and a normalised probe batch."""
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    probe = probe_norm.to(device)
    with torch.no_grad():
        s3 = torch.softmax(m(probe), dim=1).mean(0)[TARGET].item()
        feats = _penultimate(m, probe)
        fa = F.cosine_similarity(feats.mean(0).unsqueeze(0),
                                 m.linear1.weight[TARGET].unsqueeze(0)).item()
    return s3, fa


def load_state(arm_tag, rnd):
    """arm_tag: 'checker', 'clean', 'patch' — matches subdir and trigger in filename."""
    trigger = 'patch' if arm_tag == 'patch' else 'checker'
    path = os.path.join(base_dir, arm_tag,
                        f'step5_fedavg_{trigger}_a0.5_state_rnd{rnd:04d}.pt')
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu', weights_only=False)['global_state']


# ── Build fixed probes ────────────────────────────────────────────────────────
_gray_raw = 0.5 * torch.ones(32, 3, 32, 32)
_gray_norm = (_gray_raw - MEAN_T.cpu()) / STD_T.cpu()
probe_checker_trig = tu.apply_additive_norm(_gray_norm.to(device),
                                             tu.make_raw_delta('checker')).cpu()

probe_pixel50 = torch.load(pixel50_path, map_location='cpu', weights_only=True)

PROBES = [('checker_trigger', probe_checker_trig),
          ('pixel_ep50',      probe_pixel50)]

# ── Print table ───────────────────────────────────────────────────────────────
for probe_name, probe in PROBES:
    print(f'\n{"="*90}')
    print(f'Fixed probe: {probe_name}  (shape={list(probe.shape)}, device={device})')
    print(f'{"="*90}')
    print(f'{"Rnd":>5}  {"s3_chk":>8}  {"s3_cln":>8}  {"gap_chk":>8}  {"fa_chk":>7}'
          f'  |  {"s3_patch":>8}  {"gap_pat":>8}  {"fa_pat":>7}')
    print('-' * 78)

    for rnd in ROUNDS:
        s_chk   = load_state('checker', rnd)
        s_cln   = load_state('clean',   rnd)
        s_pat   = load_state('patch',   rnd)

        def f(v): return f'{v:.4f}' if v is not None else '  —   '

        if s_chk is None and s_cln is None:
            print(f'  {rnd:4d}  (checker+clean states not yet saved)'); continue

        s3_chk, fa_chk = eval_probe(s_chk, probe) if s_chk else (None, None)
        s3_cln, _      = eval_probe(s_cln, probe) if s_cln else (None, None)
        gap_chk = s3_chk - s3_cln if s3_chk and s3_cln else None

        if s_pat:
            s3_pat, fa_pat = eval_probe(s_pat, probe)
            gap_pat = s3_pat - s3_cln if s3_cln else None
            pat_str = f'{f(s3_pat):>8}  {f(gap_pat):>8}  {f(fa_pat):>7}'
        else:
            pat_str = f'{"—":>8}  {"—":>8}  {"—":>7}'

        print(f'  {rnd:4d}  {f(s3_chk):>8}  {f(s3_cln):>8}  {f(gap_chk):>8}  '
              f'{f(fa_chk):>7}  |  {pat_str}')

print('\nRead-out:')
print('  gap_chk collapses across rounds (probe fixed) → H confirmed: decay is model-driven')
print('  gap_chk flat                                  → H refuted')
print('  gap_patch holds while gap_chk collapses       → checker-specific, control valid')
