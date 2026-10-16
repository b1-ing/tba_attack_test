#!/usr/bin/env python3
"""Corrected Check 1 + Check 2 benign-prior extension.

Measures three quantities per client model at each poison rate:
  - prior    : mean target-class softmax on CLEAN held-out images (no trigger)
  - local_asr: fraction of triggered test images classified as target class
  - s3_score : mean target-class softmax on GAN probe images (what the detector sees)

Uses the existing FilterFL code unchanged; trains one adversarial client per rate
from a common warm-up checkpoint, so all rate comparisons start from the same point.

Check 2 extension: identifies the top-flagged benign clients from an existing
probe.npz file, trains them benignly from the same checkpoint, and reports their
prior — if comparable to adversarial priors, the conflation is complete.

Usage (run from repo root, with venv active):
    python jobs/check1_prior.py \\
        --code-dir FilterFL \\
        --probe-npz ~/ffl_probe/s1_ours/probes.npz \\
        --out results/analysis/check1_prior \\
        --warmup-rounds 50 \\
        --rates 0.1 0.2 0.25 0.3 0.4 0.5

The --warmup-rounds checkpoint is saved to --out/warmup_ckpt.pt so subsequent
runs reuse it. Pass --skip-warmup to reuse without retraining.
"""
import argparse
import copy
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--out', default='results/analysis/check1_prior')
ap.add_argument('--probe-npz', default=None,
                help='Existing probe.npz (rate=0.3) for Check-2 benign-prior analysis')
ap.add_argument('--warmup-rounds', type=int, default=50,
                help='Clean FL rounds before branching per-rate experiments')
ap.add_argument('--skip-warmup', action='store_true',
                help='Load existing warmup_ckpt.pt instead of retraining')
ap.add_argument('--rates', type=float, nargs='+', default=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5])
ap.add_argument('--n-benign', type=int, default=5,
                help='Number of top-flagged benign clients to evaluate for Check 2')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--gan-epochs1', type=int, default=10)
ap.add_argument('--gan-epochs2', type=int, default=5)
ap.add_argument('--adv-clients', type=int, nargs='+', default=None,
                help='Explicit adversarial client IDs to test (default: first in namelist)')
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import datetime
import numpy as np
import torch
import yaml

import algorithm
import main as ref_main
import train as train_mod
import test as test_mod
from gan import gan_defense
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open(args.params) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['seed'] = args.seed
params['dirichlet_alpha'] = args.alpha
params['defence_method'] = 'ours'
params['gan_epochs1'] = args.gan_epochs1
params['gan_epochs2'] = args.gan_epochs2
ref_main.seed(params['seed'])

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
                params=params, name=params.get('name', format(params['type'])))
handle.load_data()
handle.create_model()
print(f'adversarial_namelist: {sorted(handle.adversarial_namelist)}', flush=True)

TARGET = handle.target  # class 0


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_prior_and_asr(model, test_loader):
    """Return (prior, local_asr) for a client model.
    prior     = mean softmax[TARGET] on clean test images
    local_asr = fraction of triggered test images predicted as TARGET
    """
    model.eval()
    prior_sum, asr_correct, n = 0.0, 0, 0
    poison_positions = params.get('0_poison_pattern', [])
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            out = torch.softmax(model(data), dim=1)
            prior_sum += out[:, TARGET].sum().item()

            # Apply trigger pixel pattern
            triggered = data.clone()
            for pos in poison_positions:
                triggered[:, :, pos[0], pos[1]] = 1.0
            out_t = torch.softmax(model(triggered), dim=1)
            asr_correct += (out_t.argmax(dim=1) == TARGET).sum().item()
            n += len(data)

    return prior_sum / n, asr_correct / n


def measure_stage3(global_state, local_state):
    """Run the GAN probe and return mean target-class prob on probe images."""
    g_imgs = gan_defense(global_state, local_state, params, the_epoch=1)
    probe_model = ResNet8(num_classes=10).to(device)
    probe_model.load_state_dict(local_state)
    probe_model.eval()
    with torch.no_grad():
        out = torch.softmax(probe_model(g_imgs[0]), dim=1)
    return out.mean(0)[TARGET].item()


def train_client(global_state, client_id, poison=False, poison_rate=0.0):
    """Train one client for local_epochs (or poison_epochs) from global_state.
    Returns the client model state_dict.
    """
    p = copy.deepcopy(params)
    if poison:
        p['poison_rate'] = poison_rate
    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    dataset = handle.train_data[client_id]
    w = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id,
        params=p,
        model=model,
        dataset=dataset,
        handle=handle,
        poison=poison,
    )
    return w


# ---------------------------------------------------------------------------
# Warm-up: N rounds of clean FL to get a reasonable global model
# ---------------------------------------------------------------------------

ckpt_path = os.path.join(args.out, 'warmup_ckpt.pt')
if args.skip_warmup and os.path.exists(ckpt_path):
    print(f'Loading warm-up checkpoint from {ckpt_path}', flush=True)
    handle.model.load_state_dict(torch.load(ckpt_path, map_location=device))
elif args.warmup_rounds > 0:
    print(f'Warm-up: {args.warmup_rounds} clean FL rounds...', flush=True)
    import random as _random
    orig_is_poison = params['is_poison']
    params['is_poison'] = False  # benign-only warm-up
    for rnd in range(1, args.warmup_rounds + 1):
        clients = _random.sample(handle.benign_namelist, params['num_clients'])
        ori_w = handle.model.state_dict()
        w_locals = []
        for c in clients:
            w_locals.append(train_client(ori_w, c, poison=False))
        # Simple FedAvg
        w_avg = {}
        for k in ori_w:
            if ori_w[k].dtype != torch.int64:
                w_avg[k] = torch.mean(torch.stack([wl[k].float() for wl in w_locals]), dim=0)
            else:
                w_avg[k] = ori_w[k]
        handle.model.load_state_dict(w_avg)
        if rnd % 10 == 0:
            acc = test_mod.normal_test(handle.model, handle.test_data, params, handle, poison=False)
            print(f'  round {rnd}/{args.warmup_rounds}  clean acc={acc:.1f}%', flush=True)
    params['is_poison'] = orig_is_poison
    torch.save(handle.model.state_dict(), ckpt_path)
    print(f'Warm-up checkpoint saved to {ckpt_path}', flush=True)
else:
    print('Starting from random init (no warm-up).', flush=True)

global_state_ref = copy.deepcopy(handle.model.state_dict())

# Clean test loader for prior/ASR measurement (reuse handle's test loader,
# but we only need non-target images for ASR denominator; using full test set is fine)
test_loader = handle.test_data

# ---------------------------------------------------------------------------
# Check 1 (corrected): train one adversarial client at each rate
# ---------------------------------------------------------------------------

print('\n=== CHECK 1: Prior / Local ASR / Stage-3 per poison rate ===')
print(f'{"rate":>6}  {"prior":>7}  {"local_ASR":>10}  {"s3_score":>9}  '
      f'{"prior-s3_gap":>13}  client_id')

rows = []
adv_namelist = sorted(handle.adversarial_namelist)
if args.adv_clients is not None:
    # filter to only those that are actually adversarial
    test_clients = [c for c in args.adv_clients if c in adv_namelist]
    if not test_clients:
        print(f'WARNING: none of {args.adv_clients} are in adversarial_namelist {adv_namelist}')
        test_clients = [adv_namelist[0]]
else:
    test_clients = [adv_namelist[0]]

for adv_client in test_clients:
    print(f'\n--- Adversarial client {adv_client} ---')
    for rate in sorted(args.rates):
        orig_rate = params['poison_rate']
        params['poison_rate'] = rate

        local_state = train_client(global_state_ref, adv_client, poison=True, poison_rate=rate)

        eval_model = ResNet8(num_classes=10).to(device)
        eval_model.load_state_dict(local_state)
        prior, local_asr = measure_prior_and_asr(eval_model, test_loader)

        print(f'  Computing Stage-3 for rate={rate:.2f}...', flush=True)
        s3 = measure_stage3(global_state_ref, local_state)

        params['poison_rate'] = orig_rate

        print(f'{rate:>6.2f}  {prior:>7.4f}  {local_asr:>10.4f}  {s3:>9.4f}  '
              f'{prior - s3:>13.4f}  client={adv_client}')
        rows.append({'rate': rate, 'prior': prior, 'local_asr': local_asr, 's3': s3,
                     'client': adv_client, 'type': 'adversarial'})

# Spearman correlations
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])

adv_rows = [r for r in rows if r['type'] == 'adversarial']

# Per-client tables
for c in test_clients:
    c_rows = [r for r in adv_rows if r['client'] == c]
    print(f'\n--- Client {c} summary ---')
    print(f'{"rate":>6}  {"prior":>7}  {"local_ASR":>10}  {"s3_score":>9}')
    for r in sorted(c_rows, key=lambda x: x['rate']):
        print(f'{r["rate"]:>6.2f}  {r["prior"]:>7.4f}  {r["local_asr"]:>10.4f}  {r["s3"]:>9.4f}')

# Pooled correlations across all clients
rates_arr  = [r['rate']      for r in adv_rows]
priors_arr = [r['prior']     for r in adv_rows]
asrs_arr   = [r['local_asr'] for r in adv_rows]
s3_arr     = [r['s3']        for r in adv_rows]

print(f'\nPooled Spearman across {len(test_clients)} client(s) (n={len(adv_rows)}):')
print(f'Spearman(rate, prior)     = {spearman(rates_arr, priors_arr):+.3f}')
print(f'Spearman(rate, local_ASR) = {spearman(rates_arr, asrs_arr):+.3f}')
print(f'Spearman(rate, s3_score)  = {spearman(rates_arr, s3_arr):+.3f}')
print(f'Spearman(prior, s3_score) = {spearman(priors_arr, s3_arr):+.3f}  [should be strongly positive]')
print(f'Spearman(ASR,   s3_score) = {spearman(asrs_arr,  s3_arr):+.3f}  [should be weaker]')

# ---------------------------------------------------------------------------
# Check 2 benign extension: measure prior for top-flagged benign clients
# ---------------------------------------------------------------------------

if args.probe_npz and os.path.exists(args.probe_npz):
    print(f'\n=== CHECK 2 EXTENSION: Benign client prior from {args.probe_npz} ===')
    d = np.load(args.probe_npz, allow_pickle=True)
    p0_all, is_adv, clients_all = d['probs'][:, 0], d['is_adv'], d['client']
    std_all = d['standard']
    flagged_ben_mask = (~is_adv) & (p0_all > std_all)
    ben_clients_flagged = clients_all[flagged_ben_mask]

    if len(ben_clients_flagged) == 0:
        print('  No flagged benign clients found in probe data.')
    else:
        # Rank benign clients by how often they were flagged
        from collections import Counter
        flag_counts = Counter(ben_clients_flagged.tolist())
        top_ben = [c for c, _ in flag_counts.most_common(args.n_benign)]
        print(f'  Top {len(top_ben)} most-flagged benign clients: {top_ben}')
        print(f'  (flag counts: {[flag_counts[c] for c in top_ben]})')

        # Reference adversarial prior at rate=0.3 from rows above
        ref_adv_row = next((r for r in rows if abs(r['rate'] - 0.3) < 0.01), None)
        if ref_adv_row:
            print(f'  Reference adversarial prior (rate=0.3): {ref_adv_row["prior"]:.4f}')

        print(f'\n  {"client":>8}  {"prior":>7}  {"flag_count":>12}  {"mean_s3_score":>14}')
        for ben_c in top_ben:
            # Train benignly from warm-up checkpoint
            local_state = train_client(global_state_ref, ben_c, poison=False)
            eval_model = ResNet8(num_classes=10).to(device)
            eval_model.load_state_dict(local_state)
            prior_ben, _ = measure_prior_and_asr(eval_model, test_loader)

            # Mean Stage-3 score for this benign client from probe data
            c_mask = clients_all == ben_c
            mean_s3 = p0_all[c_mask].mean() if c_mask.any() else float('nan')

            print(f'  {ben_c:>8}  {prior_ben:>7.4f}  {flag_counts[ben_c]:>12}  {mean_s3:>14.4f}')
            rows.append({'rate': float('nan'), 'prior': prior_ben, 'local_asr': float('nan'),
                         's3': mean_s3, 'client': ben_c, 'type': 'benign_flagged'})

        # Summary: benign vs adversarial prior at rate=0.3
        adv_priors = [r['prior'] for r in rows if r['type'] == 'adversarial' and abs(r['rate'] - 0.3) < 0.05]
        ben_priors = [r['prior'] for r in rows if r['type'] == 'benign_flagged']
        if adv_priors and ben_priors:
            print(f'\n  Adversarial prior (rate≈0.3): mean={np.mean(adv_priors):.4f}')
            print(f'  Flagged benign prior:          mean={np.mean(ben_priors):.4f}')
            gap = np.mean(adv_priors) - np.mean(ben_priors)
            print(f'  Gap: {gap:+.4f}  [near-zero = conflation confirmed]')
else:
    print('\n(No --probe-npz supplied; skipping Check-2 benign extension.)')

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

import csv
out_csv = os.path.join(args.out, 'check1_prior.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['type', 'client', 'rate', 'prior', 'local_asr', 's3'])
    w.writeheader()
    w.writerows(rows)
print(f'\nResults saved to {out_csv}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    adv_rows = [r for r in rows if r['type'] == 'adversarial']
    adv_rows.sort(key=lambda r: r['rate'])
    rates_p = [r['rate']      for r in adv_rows]
    priors_p = [r['prior']    for r in adv_rows]
    asrs_p  = [r['local_asr'] for r in adv_rows]
    s3_p    = [r['s3']        for r in adv_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: prior and Stage-3 vs rate
    axes[0].plot(rates_p, priors_p, 'o-', color='tab:blue',   label='prior (clean test)')
    axes[0].plot(rates_p, s3_p,     's--', color='tab:orange', label='Stage-3 score')
    axes[0].set(xlabel='poison rate', ylabel='target-class softmax', title='Prior vs Stage-3 score')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Right: prior and ASR vs rate (normalized so both start at 0)
    prior_norm = np.array(priors_p) - priors_p[0]
    asr_norm   = np.array(asrs_p)   - asrs_p[0]
    axes[1].plot(rates_p, prior_norm, 'o-', color='tab:blue',  label='prior (offset)')
    axes[1].plot(rates_p, asr_norm,   's-', color='tab:red',   label='local ASR (offset)')
    axes[1].axhline(0, color='k', ls=':', lw=0.8)
    axes[1].set(xlabel='poison rate', ylabel='Δ from lowest rate',
                title='Prior vs ASR rate-sensitivity\n(both offset to 0 at lowest rate)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # If benign priors available, add them to left plot
    ben_rows = [r for r in rows if r['type'] == 'benign_flagged']
    if ben_rows:
        ben_priors_p = [r['prior'] for r in ben_rows]
        axes[0].axhline(np.mean(ben_priors_p), color='tab:green', ls='--', lw=1.5,
                        label=f'flagged benign prior (n={len(ben_priors_p)})')
        axes[0].legend()

    fig.suptitle(f'Check 1 — Prior / Stage-3 / ASR  (warmup={args.warmup_rounds} rounds, seed={args.seed})')
    fig.tight_layout()
    out_png = os.path.join(args.out, 'check1_prior.png')
    fig.savefig(out_png, dpi=150)
    print(f'Plot saved to {out_png}')
    plt.close(fig)
except ImportError:
    print('(matplotlib not available, skipping plot)')

print('\ndone', flush=True)
