"""Experiment H (v3) — correct form: does FilterFL's own s3 decay with the probe frozen?

Probe = pixel-space probe generated at round 50 from the FilterFL t1 run (already saved).
Frozen once; NOT re-optimised per round.

For each round in ROUNDS, evaluates that fixed probe on d1 (the adversarial client's local model
at that round) from the FilterFL defended run. d1 is what FilterFL Stage 3 actually scores.

Metrics:
  s3  = softmax(d1_model(frozen_probe)).mean(0)[TARGET]   (FilterFL's actual threshold quantity)
  fa  = cosine(mean_penultimate(d1_model, frozen_probe), w[TARGET])

Read-out (per spec):
  frozen-probe s3 falls across rounds (toward rho) -> H SUPPORTED: decay is model-driven
  frozen-probe s3 stays flat               -> H not supported; rethink

Usage (checker arm — data already exists from probe-spectrum run):
  python3 step5_Htest_analysis_v3.py \\
      <probe_dir>    ~/ffl_step5/probes_checker   (or probes_patch for patch arm) \\
      <filterfl_dir> ~/tba_attack_test/FilterFL   \\
      <arm_label>    checker                       (printed in table)

For patch arm, run again once patch probe-spectrum job completes:
  python3 step5_Htest_analysis_v3.py ~/ffl_step5/probes_patch ~/tba_attack_test/FilterFL patch
"""

import sys, os, glob, re
import torch
import torch.nn.functional as F

probe_dir   = sys.argv[1]
ffl_dir     = sys.argv[2]
arm_label   = sys.argv[3] if len(sys.argv) > 3 else 'unknown'

sys.path.insert(0, ffl_dir)
import trigger_utils as tu
from models.ResNet8 import ResNet8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TARGET = 0
ROUNDS = [50, 150, 250, 330, 400]

print(f'Arm: {arm_label}  probe_dir: {probe_dir}', flush=True)
print(f'trigger_utils v{tu.VERSION}  device: {device}', flush=True)

# ── Load frozen probe (saved at ep 50) ───────────────────────────────────────
probe_path = os.path.join(probe_dir, 'pixel_probe_ep0050.pt')
assert os.path.exists(probe_path), f'Frozen probe not found: {probe_path}'
frozen_probe = torch.load(probe_path, map_location='cpu', weights_only=True).to(device)
print(f'Frozen probe loaded: shape={list(frozen_probe.shape)}  (ep=50, NOT re-optimised)',
      flush=True)

# ── Model helpers ─────────────────────────────────────────────────────────────
def _penultimate(model, x):
    """ResNet8 representation after avgpool, before linear1."""
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def eval_frozen_probe(state_dict, probe):
    """Load state into ResNet8, evaluate frozen probe, return (s3, fa).

    Uses train() mode to match algorithm.py Stage 3 exactly — ResNet8_model
    is never set to eval() there, so BN uses batch statistics from the probe.
    s3 = torch.max(softmax(...).mean(0)) with argmax==0 check, matching the
    actual FilterFL detection signal (not class-0 specifically).
    """
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    m.train()   # match algorithm.py: BN uses batch stats from probe, not running stats
    with torch.no_grad():
        out = torch.softmax(m(probe), dim=1).mean(0)
        out_val, out_n = torch.max(out, dim=0)
        s3 = out_val.item() if out_n.item() == TARGET else 0.0
        feats = _penultimate(m, probe)
        fa = F.cosine_similarity(feats.mean(0).unsqueeze(0),
                                 m.linear1.weight[TARGET].unsqueeze(0)).item()
    return s3, fa


# ── Evaluate across rounds ────────────────────────────────────────────────────
print(f'\n{"Rnd":>5}  {"s3 (frozen probe)":>18}  {"fa":>8}  note')
print('-' * 55)

results = []
for rnd in ROUNDS:
    # adv_local_ep{N}.pt = adversarial client local model saved by algorithm.py Stage 3 loop.
    # This is the model FilterFL actually thresholds (w_locals[i]), not ori_weight/w_glob.
    mf = os.path.join(probe_dir, f'adv_local_ep{rnd:04d}.pt')
    if not os.path.exists(mf):
        print(f'  {rnd:4d}  (adv_local file not found: {mf})')
        results.append((rnd, None, None))
        continue

    state = torch.load(mf, map_location='cpu', weights_only=False)

    s3, fa = eval_frozen_probe(state, frozen_probe)
    note = 'probe epoch' if rnd == 50 else ''
    print(f'  {rnd:4d}  {s3:>18.4f}  {fa:>8.4f}  {note}', flush=True)
    results.append((rnd, s3, fa))

# ── Summary ───────────────────────────────────────────────────────────────────
valid = [(r, s3, fa) for r, s3, fa in results if s3 is not None]
if len(valid) >= 2:
    s3_first = valid[0][1]
    s3_last  = valid[-1][1]
    fa_first = valid[0][2]
    fa_last  = valid[-1][2]
    print(f'\nSummary ({arm_label}):')
    print(f'  s3  rnd{valid[0][0]}={s3_first:.4f}  rnd{valid[-1][0]}={s3_last:.4f}'
          f'  delta={s3_last - s3_first:+.4f}')
    print(f'  fa  rnd{valid[0][0]}={fa_first:.4f}  rnd{valid[-1][0]}={fa_last:.4f}'
          f'  delta={fa_last - fa_first:+.4f}')

print('\nRead-out:')
print('  s3 falls toward rho (~0.2) with probe frozen -> H SUPPORTED: decay is model-driven')
print('  s3 flat                                      -> H not supported')
