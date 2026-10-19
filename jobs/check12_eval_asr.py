#!/usr/bin/env python3
"""Post-run ASR evaluation for check12 (checker trigger).

The main.py eval applies the patch trigger to the test set regardless of trigger_type,
so Poison:True accuracy in the check12 log is measuring the wrong trigger.
This script loads the final saved model from check12, applies the CHECKER trigger
to the non-target-class test set, and measures true ASR.

Also measures clean accuracy for completeness.
"""
import argparse, copy, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',      required=True, help='FilterFL scratch or source dir')
ap.add_argument('--checkpoint',    required=True, help='Path to saved model (.pt or .pth)')
ap.add_argument('--amplitude',     type=float, default=0.50)
ap.add_argument('--alpha',         type=float, default=0.5)
ap.add_argument('--seed',          type=int,   default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)

import torch
import yaml

import main as ref_main
import train as train_mod
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['seed']              = args.seed
params['dirichlet_alpha']   = args.alpha
params['defence_method']    = 'ours'
params['trigger_type']      = 'checker'
params['trigger_amplitude'] = args.amplitude
ref_main.seed(params['seed'])

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET = handle.target
print(f'Target class: {TARGET}', flush=True)

model = ResNet8(num_classes=10).to(device)
ckpt = torch.load(args.checkpoint, map_location=device)
# Handle both raw state_dict and wrapped checkpoints
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    model.load_state_dict(ckpt['state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f'Loaded checkpoint: {args.checkpoint}', flush=True)


def apply_checker(imgs):
    """Apply additive checker trigger to a batch (C,H,W)."""
    p = {'trigger_type': 'checker', 'trigger_amplitude': args.amplitude}
    additive = train_mod._get_additive_trigger('checker', p, imgs[0].shape, imgs.device)
    return imgs + additive.unsqueeze(0)


# ── Clean accuracy ────────────────────────────────────────────────────────────
correct_clean, total_clean = 0, 0
with torch.no_grad():
    for imgs, lbls in handle.test_data:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs).argmax(dim=1)
        correct_clean += preds.eq(lbls).sum().item()
        total_clean   += len(lbls)
clean_acc = 100.0 * correct_clean / total_clean
print(f'\nClean accuracy: {correct_clean}/{total_clean} ({clean_acc:.2f}%)', flush=True)

# ── ASR: checker trigger on non-target-class test images ────────────────────
# ASR = fraction of triggered non-target images classified as TARGET
correct_asr, total_asr = 0, 0
with torch.no_grad():
    for imgs, lbls in handle.test_data:
        # Keep only non-target images
        mask = lbls != TARGET
        if not mask.any():
            continue
        imgs_nt = imgs[mask].to(device)
        triggered = apply_checker(imgs_nt)
        preds = model(triggered).argmax(dim=1)
        correct_asr += (preds == TARGET).sum().item()
        total_asr   += len(imgs_nt)
asr = 100.0 * correct_asr / total_asr if total_asr > 0 else 0.0
print(f'ASR (checker trigger): {correct_asr}/{total_asr} ({asr:.2f}%)', flush=True)

# ── Mean softmax p(target) on triggered images ───────────────────────────────
total_p, n_batches = 0.0, 0
with torch.no_grad():
    for imgs, lbls in handle.test_data:
        mask = lbls != TARGET
        if not mask.any():
            continue
        imgs_nt = imgs[mask].to(device)
        triggered = apply_checker(imgs_nt)
        probs = torch.softmax(model(triggered), dim=1)
        total_p  += probs[:, TARGET].mean().item()
        n_batches += 1
mean_p_target = total_p / n_batches if n_batches > 0 else 0.0
print(f'Mean p(target) on triggered: {mean_p_target:.4f}', flush=True)

print(f'\n=== Summary ===')
print(f'  Clean accuracy : {clean_acc:.2f}%')
print(f'  ASR (checker)  : {asr:.2f}%')
print(f'  Mean p(target) : {mean_p_target:.4f}')
