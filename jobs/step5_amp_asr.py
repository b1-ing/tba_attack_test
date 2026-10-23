#!/usr/bin/env python3
"""ASR evaluation for amplitude isolation runs.

For each saved global model (model_ep{N}.pt):
  - Apply checker trigger at the run's amplitude to CIFAR-10 test non-target images
  - Compute ASR = fraction of triggered images classified as TARGET (eval mode)
  - Also compute clean accuracy on target-class images (sanity check)

Designed for the amp_isolation sweep (eps16/eps32/eps64) and can also be run
on eps4/eps8 and checker arms using their probe dirs.
"""
import argparse, os, sys, pickle
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--trigger',   required=True,
                help='trigger name: eps4/eps8/eps16/eps32/eps64/checker/checker_em')
ap.add_argument('--probe-dir', required=True, help='dir with model_ep*.pt and adv_local_ep*.pt')
ap.add_argument('--ffl-dir',   required=True, help='FilterFL source dir for models/ResNet8.py')
ap.add_argument('--data-dir',  required=True, help='cifar-10-batches-py dir')
ap.add_argument('--out',       required=True, help='output .txt path')
ap.add_argument('--rounds',    nargs='+', type=int, default=[50, 150, 400, 600],
                help='rounds to evaluate (looks for model_ep{N:04d}.pt)')
ap.add_argument('--rho',       type=float, default=0.5, help='detection threshold')
args = ap.parse_args()

sys.path.insert(0, args.ffl_dir)
from models.ResNet8 import ResNet8

TARGET = 0
device = torch.device('cpu')

# ── amplitude map ──────────────────────────────────────────────────────────────
AMP_MAP = {
    'eps4':              4.0 / 255.0,
    'eps8':              8.0 / 255.0,
    'eps16':            16.0 / 255.0,
    'eps32':            32.0 / 255.0,
    'eps64':            64.0 / 255.0,
    'checker':          0.1145,
    'checker_em':       3.268 / (3 * 32 * 32) ** 0.5,
}

trig_key = args.trigger.lower().replace('-', '_')
if trig_key not in AMP_MAP:
    raise ValueError(f'Unknown trigger for amplitude lookup: {args.trigger}')
amplitude = AMP_MAP[trig_key]
print(f'Trigger: {args.trigger}  amplitude (raw): {amplitude:.5f}  ({amplitude*255:.2f}/255)',
      flush=True)

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
def load_cifar_test(data_dir, n_target=200, n_nontarget=200):
    path = os.path.join(data_dir, 'test_batch')
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'])
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:,None,None]
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:,None,None]
    def norm(x): return torch.from_numpy((x - mean) / std)
    ti = np.where(labels == TARGET)[0][:n_target]
    ni = np.where(labels != TARGET)[0][:n_nontarget]
    return norm(imgs[ti]), norm(imgs[ni]), labels[ti], labels[ni]

clean_target, clean_nontarget, _, _ = load_cifar_test(args.data_dir)

# ── apply checker trigger at amplitude ────────────────────────────────────────
sys.path.insert(0, args.ffl_dir)
import trigger_utils as tu

hw = np.array([((r + c) % 2) * 2.0 - 1.0
               for r in range(32) for c in range(32)],
              dtype=np.float32).reshape(32, 32)
delta_raw = np.stack([hw * amplitude] * 3, axis=0)   # (3,32,32)
triggered_nontarget = tu.apply_additive_norm(clean_nontarget, delta_raw)

# ── model helpers ──────────────────────────────────────────────────────────────
def load_state_dict_from(path):
    """Load state dict from either adv_local format (flat) or model_ep format ({'d2': ...})."""
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(raw, dict) and 'd2' in raw:
        return raw['d2']   # global model after FilterFL aggregation
    return raw             # adv_local format: flat state dict

def load_model_eval(path):
    sd = load_state_dict_from(path)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m

def asr(m, x_triggered):
    with torch.no_grad():
        logits = m(x_triggered)
        preds = logits.argmax(dim=1)
    return (preds == TARGET).float().mean().item()

def clean_acc(m, x_clean, labels_val=TARGET):
    with torch.no_grad():
        logits = m(x_clean)
        preds = logits.argmax(dim=1)
    return (preds == labels_val).float().mean().item()

def s3_train(state_dict, probe):
    """s3 in train mode from a flat state dict, matching FilterFL Stage 3."""
    m2 = ResNet8(num_classes=10).to(device)
    m2.load_state_dict(state_dict)
    # stays train mode (no .eval()) — matches algorithm.py Stage 3
    with torch.no_grad():
        out = torch.softmax(m2(probe), dim=1).mean(0)
    return out[TARGET].item()

# ── per-round evaluation ───────────────────────────────────────────────────────
results = []
probe_path = os.path.join(args.probe_dir, 'pixel_probe_ep0050.pt')
has_probe  = os.path.exists(probe_path)
if has_probe:
    frozen_probe = torch.load(probe_path, map_location='cpu', weights_only=False)
    if frozen_probe.dim() == 3:
        frozen_probe = frozen_probe.unsqueeze(0)

for rnd in args.rounds:
    model_path  = os.path.join(args.probe_dir, f'model_ep{rnd:04d}.pt')
    adv_path    = os.path.join(args.probe_dir, f'adv_local_ep{rnd:04d}.pt')

    if not os.path.exists(model_path):
        results.append({'rnd': rnd, 'missing': True})
        print(f'  r{rnd}: no global model at {model_path}', flush=True)
        continue

    # ASR: global model in eval mode (attack effectiveness on federated model)
    m_global = load_model_eval(model_path)
    a  = asr(m_global, triggered_nontarget)
    ca = clean_acc(m_global, clean_target)

    # s3: adversarial LOCAL model in train mode (FilterFL detection criterion)
    s3_val = float('nan')
    below_rho = None
    if has_probe and os.path.exists(adv_path):
        adv_sd = torch.load(adv_path, map_location='cpu', weights_only=False)
        s3_val = s3_train(adv_sd, frozen_probe)
        below_rho = s3_val < args.rho

    results.append({
        'rnd': rnd, 'asr': a, 'clean_acc': ca,
        's3': s3_val, 'below_rho': below_rho,
    })
    br_str = ''
    if below_rho is True:  br_str = 'adv_local evades'
    elif below_rho is False: br_str = 'adv_local caught'
    print(f'  r{rnd}: ASR={a:.3f}  clean_acc={ca:.3f}  '
          f's3(adv_local)={s3_val:.4f}  {br_str}', flush=True)

# ── output ────────────────────────────────────────────────────────────────────
lines = [
    f'trigger: {args.trigger}',
    f'amplitude (raw): {amplitude:.5f}  ({amplitude*255:.2f}/255)',
    f'rho: {args.rho}',
    '',
    f'  {"Rnd":>5}  {"ASR(global)":>12}  {"clean_acc":>10}  {"s3(adv_local)":>14}  note',
    '  ' + '-'*62,
]
for r in results:
    if r.get('missing'):
        lines.append(f'  {r["rnd"]:>5}  (no file)')
    else:
        note = ''
        if r['below_rho'] is True:
            note = 'evades'
        elif r['below_rho'] is False:
            note = 'caught'
        lines.append(f'  {r["rnd"]:>5}  {r["asr"]:>12.3f}  {r["clean_acc"]:>10.3f}  '
                     f'{r["s3"]:>14.4f}  {note}')

out_text = '\n'.join(lines)
print('\n' + out_text, flush=True)
with open(args.out, 'w') as f:
    f.write(out_text + '\n')
print(f'\nSaved to {args.out}', flush=True)
