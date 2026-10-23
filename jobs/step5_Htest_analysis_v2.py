"""Experiment H (v2) analysis — discriminative probe.

Probe = trigger applied to ALL non-target CIFAR-10 test images (classes 1-9),
loaded from pickle and frozen before any model evaluation.

For each round in [50, 150, 250, 330, 400]:
  - checker arm (backdoored FedAvg) vs clean arm (no poison)
  - patch arm (backdoored FedAvg)   vs clean arm

Metrics per (round, trigger):
  rr_adv   = fraction of probe images backdoored model routes to TARGET (class 0)
  rr_clean = same, clean model
  gap      = rr_adv - rr_clean
  margin   = mean(logit_target - logit_true_class) on backdoored model
  fa       = cosine(mean_penultimate_feats, w[TARGET]) on backdoored model

Read-out:
  margin/fa decays for checker, holds for patch -> H supported (checker-specific decay)
  both flat -> H not supported by this test
  rr saturated at 1.0 throughout -> use margin and fa; rate can't capture the signal

Usage:
  python3 step5_Htest_analysis_v2.py <Htest_base_dir> <filterfl_dir> <cifar10_data_dir>

  Htest_base_dir  parent of checker/, clean/, patch/ subdirs from step5_Htest_train.sh
  filterfl_dir    path to FilterFL/ (trigger_utils, ResNet8)
  cifar10_data_dir path to cifar-10-batches-py/ directory (contains test_batch)
"""

import sys, os, pickle
import numpy as np
import torch
import torch.nn.functional as F

base_dir    = sys.argv[1]
ffl_dir     = sys.argv[2]
cifar10_dir = sys.argv[3]

sys.path.insert(0, ffl_dir)
import trigger_utils as tu
from models.ResNet8 import ResNet8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TARGET = 0
ROUNDS = [50, 150, 250, 330, 400]

PATCH_POSITIONS = [[0,0],[0,1],[0,2],[0,3],[0,6],[0,7],[0,8],[0,9],
                   [3,0],[3,1],[3,2],[3,3],[3,6],[3,7],[3,8],[3,9]]
AMP_RAW = 0.1145

MEAN_T = torch.tensor(tu.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
STD_T  = torch.tensor(tu.STD,  dtype=torch.float32).view(1, 3, 1, 1)

# ── Load CIFAR-10 test set from pickle ───────────────────────────────────────
test_batch_path = os.path.join(cifar10_dir, 'test_batch')
with open(test_batch_path, 'rb') as f:
    d = pickle.load(f, encoding='bytes')

raw_all = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0   # (10000,3,32,32)
lbl_all = np.array(d[b'labels'])

non_target_mask = lbl_all != TARGET
raw_nontarget   = torch.from_numpy(raw_all[non_target_mask])              # (N,3,32,32) in [0,1]
true_lbl_cpu    = torch.from_numpy(lbl_all[non_target_mask]).long()       # (N,) true labels

N = len(raw_nontarget)
print(f'Non-target test images: {N}  (classes 1-9)', flush=True)
print(f'trigger_utils version: {tu.VERSION}', flush=True)
print(f'Device: {device}', flush=True)

# ── Normalize ────────────────────────────────────────────────────────────────
imgs_norm_cpu = (raw_nontarget - MEAN_T) / STD_T    # (N,3,32,32) normalized, on CPU

# ── Build fixed probes (once, on CPU; move to device for eval) ───────────────
delta_checker   = tu.make_raw_delta('checker', amplitude=AMP_RAW)          # (3,32,32) numpy
probe_checker   = tu.apply_additive_norm(imgs_norm_cpu, delta_checker)     # (N,3,32,32) tensor

patch_mask_cpu  = tu.get_patch_mask(PATCH_POSITIONS, device=torch.device('cpu'))
probe_patch     = tu.apply_patch_norm(imgs_norm_cpu, patch_mask_cpu, raw_value=1.0)

# Move to GPU once
probe_checker_dev = probe_checker.to(device)
probe_patch_dev   = probe_patch.to(device)
true_lbl_dev      = true_lbl_cpu.to(device)

print(f'Probe shapes: checker={list(probe_checker_dev.shape)}, '
      f'patch={list(probe_patch_dev.shape)}', flush=True)


# ── Model helpers ────────────────────────────────────────────────────────────
def _penultimate(model, x):
    """ResNet8 penultimate representation (after avgpool, before linear1)."""
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def eval_probe(state, probe, true_lbl=None):
    """Evaluate fixed probe on a model state.

    Returns (route_rate, margin, fa):
      route_rate : fraction of probe images predicted as TARGET
      margin     : mean(logit[TARGET] - logit[true_class]) if true_lbl given, else None
      fa         : cosine(mean penultimate feats, w[TARGET])
    """
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    with torch.no_grad():
        logits = m(probe)                                               # (N, 10)
        preds  = logits.argmax(dim=1)
        route_rate = (preds == TARGET).float().mean().item()

        margin = None
        if true_lbl is not None:
            c0 = logits[:, TARGET]
            tc = logits[torch.arange(len(logits), device=device), true_lbl]
            margin = (c0 - tc).mean().item()

        feats = _penultimate(m, probe)
        fa = F.cosine_similarity(feats.mean(0).unsqueeze(0),
                                 m.linear1.weight[TARGET].unsqueeze(0)).item()
    return route_rate, margin, fa


def load_state(arm_tag, rnd):
    trigger = 'patch' if arm_tag == 'patch' else 'checker'
    path = os.path.join(base_dir, arm_tag,
                        f'step5_fedavg_{trigger}_a0.5_state_rnd{rnd:04d}.pt')
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found', flush=True)
        return None
    return torch.load(path, map_location='cpu', weights_only=False)['global_state']


# ── Print table ───────────────────────────────────────────────────────────────
header = (f'{"Rnd":>5}  {"Trigger":>9}  {"rr_adv":>8}  {"rr_cln":>8}  '
          f'{"gap":>8}  {"margin":>9}  {"fa":>8}')
print(f'\n{header}')
print('-' * 76)

for rnd in ROUNDS:
    s_chk = load_state('checker', rnd)
    s_cln = load_state('clean',   rnd)
    s_pat = load_state('patch',   rnd)

    def f(v): return f'{v:+.4f}' if v is not None else '   —   '

    if s_chk is not None and s_cln is not None:
        rr_adv, mg,  fa  = eval_probe(s_chk, probe_checker_dev, true_lbl_dev)
        rr_cln, _,   _   = eval_probe(s_cln, probe_checker_dev)
        gap = rr_adv - rr_cln
        print(f'  {rnd:4d}  {"checker":>9}  {rr_adv:>8.4f}  {rr_cln:>8.4f}  '
              f'{gap:>+8.4f}  {f(mg):>9}  {fa:>+8.4f}', flush=True)

    if s_pat is not None and s_cln is not None:
        rr_adv_p, mg_p, fa_p = eval_probe(s_pat, probe_patch_dev, true_lbl_dev)
        rr_cln_p, _,    _    = eval_probe(s_cln, probe_patch_dev)
        gap_p = rr_adv_p - rr_cln_p
        print(f'  {rnd:4d}  {"patch":>9}  {rr_adv_p:>8.4f}  {rr_cln_p:>8.4f}  '
              f'{gap_p:>+8.4f}  {f(mg_p):>9}  {fa_p:>+8.4f}', flush=True)

    print(flush=True)

print('Read-out (measured):')
print('  margin/fa decays checker, holds patch -> H supported (checker-specific)')
print('  both flat                             -> H not supported')
print('  rr saturated at 1.0 throughout       -> margin and fa carry the signal')
