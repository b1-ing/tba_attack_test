#!/usr/bin/env python3
"""Check 3b: Extended UAP correlation test.

Extends check3_uap.py by:
  1. Adding N unflagged benign clients spanning the class-0 share range
  2. Reporting within-group Spearman (adversarial, benign-flagged, benign-unflagged, all-benign)
  3. Correlating class-0 data share with UAP score (causal chain: share→susceptibility→s3)

Client groups:
  - Adversarial: 4 clients [34,36,45,47] at rates [0.3, 0.5] → n=8
  - Benign flagged: top 5 from --probe-npz → n=5
  - Benign unflagged: --n-unflagged clients spanning the class-0 share range → n=20 default

    python jobs/check3b_uap_extended.py --code-dir FilterFL --out results/analysis/check3b \\
        --probe-npz ~/ffl_probe/s1_ours/probes.npz --skip-warmup
"""
import argparse
import copy
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--out', default='results/analysis/check3b')
ap.add_argument('--probe-npz', default=None)
ap.add_argument('--warmup-rounds', type=int, default=50)
ap.add_argument('--skip-warmup', action='store_true')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--rates', type=float, nargs='+', default=[0.3, 0.5])
ap.add_argument('--adv-clients', type=int, nargs='+', default=None)
ap.add_argument('--n-flagged', type=int, default=5)
ap.add_argument('--n-unflagged', type=int, default=20,
                help='Unflagged benign clients spanning the class-0 share distribution')
ap.add_argument('--uap-eps', type=float, default=0.1)
ap.add_argument('--uap-steps', type=int, default=300)
ap.add_argument('--uap-n-images', type=int, default=512)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--alpha', type=float, default=0.5)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import csv
import datetime
import numpy as np
import torch
import yaml

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
ref_main.seed(params['seed'])

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
                params=params, name=params.get('name', format(params['type'])))
handle.load_data()
handle.create_model()
TARGET = handle.target
adv_namelist = sorted(handle.adversarial_namelist)
print(f'adversarial_namelist: {adv_namelist}', flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_class0_share(client_id):
    """Fraction of training samples with label TARGET for this client."""
    _, loader = handle.train_data[client_id]
    total, count = 0, 0
    for _, labels in loader:
        count += (labels == TARGET).sum().item()
        total += len(labels)
    return count / total if total > 0 else 0.0


def train_client(global_state, client_id, poison=False, poison_rate=0.0):
    p = copy.deepcopy(params)
    if poison:
        p['poison_rate'] = poison_rate
    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    w = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id,
        params=p,
        model=model,
        dataset=handle.train_data[client_id],
        handle=handle,
        poison=poison,
    )
    return w


def measure_stage3(global_state, local_state):
    g_imgs = gan_defense(global_state, local_state, params, the_epoch=1)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    m.eval()
    with torch.no_grad():
        out = torch.softmax(m(g_imgs[0]), dim=1)
    return out.mean(0)[TARGET].item()


def compute_uap(global_model, test_loader, eps, steps, n_images):
    global_model.eval()
    batches = []
    collected = 0
    for data, _ in test_loader:
        batches.append(data)
        collected += len(data)
        if collected >= n_images:
            break
    images = torch.cat(batches)[:n_images].to(device)

    delta = torch.zeros(1, *images.shape[1:], device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.01)

    print(f'Computing UAP (eps={eps}, steps={steps}, n={len(images)})...', flush=True)
    for step in range(steps):
        opt.zero_grad()
        perturbed = (images + delta).clamp(0.0, 1.0)
        probs = torch.softmax(global_model(perturbed), dim=1)
        (-probs[:, TARGET].mean()).backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                s = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
            print(f'  step {step+1}  global_uap_score={s:.4f}', flush=True)

    with torch.no_grad():
        final_s = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
    print(f'UAP done. Global model score: {final_s:.4f}', flush=True)
    return delta.detach(), images


def measure_uap_score(client_model, images, uap_delta):
    client_model.eval()
    with torch.no_grad():
        perturbed = (images + uap_delta).clamp(0.0, 1.0)
        probs = torch.softmax(client_model(perturbed), dim=1)
    return probs[:, TARGET].mean().item()


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(set(x.tolist())) < 2:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

warmup_dir = args.warmup_dir or args.out
ckpt_path = os.path.join(warmup_dir, 'warmup_ckpt.pt')
if args.skip_warmup and os.path.exists(ckpt_path):
    print(f'Loading warm-up checkpoint from {ckpt_path}', flush=True)
    handle.model.load_state_dict(torch.load(ckpt_path, map_location=device))
elif args.warmup_rounds > 0:
    import random as _random
    print(f'Warm-up: {args.warmup_rounds} clean FL rounds...', flush=True)
    orig_is_poison = params['is_poison']
    params['is_poison'] = False
    for rnd in range(1, args.warmup_rounds + 1):
        clients = _random.sample(handle.benign_namelist, params['num_clients'])
        ori_w = handle.model.state_dict()
        w_locals = [train_client(ori_w, c, poison=False) for c in clients]
        w_avg = {}
        for k in ori_w:
            if ori_w[k].dtype != torch.int64:
                w_avg[k] = torch.mean(torch.stack([wl[k].float() for wl in w_locals]), dim=0)
            else:
                w_avg[k] = ori_w[k]
        handle.model.load_state_dict(w_avg)
        if rnd % 10 == 0:
            acc = test_mod.normal_test(handle.model, handle.test_data, params, handle, poison=False)
            print(f'  round {rnd}/{args.warmup_rounds}  acc={acc:.1f}%', flush=True)
    params['is_poison'] = orig_is_poison
    torch.save(handle.model.state_dict(), ckpt_path)

global_state_ref = copy.deepcopy(handle.model.state_dict())

# ---------------------------------------------------------------------------
# Class-0 share for ALL clients (no training needed)
# ---------------------------------------------------------------------------

print('Computing class-0 shares for all clients...', flush=True)
all_clients = list(range(params['number_of_total_participants']))
shares = {c: get_class0_share(c) for c in all_clients}
print(f'  Share range: {min(shares.values()):.3f} – {max(shares.values()):.3f}', flush=True)

# ---------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------

# Flagged benign clients from probe run
if args.probe_npz and os.path.exists(args.probe_npz):
    from collections import Counter
    d = np.load(args.probe_npz, allow_pickle=True)
    p0, is_adv, clients_col, std = d['probs'][:, 0], d['is_adv'], d['client'], d['standard']
    flagged_ben = clients_col[(~is_adv) & (p0 > std)]
    flagged_ids = [c for c, _ in Counter(flagged_ben.tolist()).most_common(args.n_flagged)]
else:
    flagged_ids = []

print(f'Flagged benign clients: {flagged_ids}', flush=True)

# Unflagged benign: span the share distribution in equal quantile buckets
benign_all = sorted(handle.benign_namelist)
unflagged_pool = [(c, shares[c]) for c in benign_all if c not in flagged_ids]
unflagged_pool.sort(key=lambda x: x[1])

# Pick --n-unflagged clients evenly spaced across the sorted list
n_pick = args.n_unflagged
if len(unflagged_pool) <= n_pick:
    unflagged_ids = [c for c, _ in unflagged_pool]
else:
    step = len(unflagged_pool) / n_pick
    unflagged_ids = [unflagged_pool[int(i * step)][0] for i in range(n_pick)]

print(f'Unflagged benign clients ({len(unflagged_ids)}): {unflagged_ids}', flush=True)
print(f'  share range of selected: '
      f'{min(shares[c] for c in unflagged_ids):.3f} – {max(shares[c] for c in unflagged_ids):.3f}',
      flush=True)

# Adversarial clients
test_adv = args.adv_clients if args.adv_clients else adv_namelist[:4]
test_adv = [c for c in test_adv if c in adv_namelist]

# ---------------------------------------------------------------------------
# Compute UAP
# ---------------------------------------------------------------------------

global_model = ResNet8(num_classes=10).to(device)
global_model.load_state_dict(global_state_ref)
uap_delta, uap_images = compute_uap(global_model, handle.test_data,
                                    eps=args.uap_eps, steps=args.uap_steps,
                                    n_images=args.uap_n_images)

# ---------------------------------------------------------------------------
# Measure all clients
# ---------------------------------------------------------------------------

rows = []

def measure_client(client_id, client_type, poison, rate):
    params['poison_rate'] = rate if poison else params.get('poison_rate', 0.3)
    local_state = train_client(global_state_ref, client_id, poison=poison, poison_rate=rate)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    uap_sc = measure_uap_score(m, uap_images, uap_delta)
    s3_sc = measure_stage3(global_state_ref, local_state)
    share = shares[client_id]
    print(f'{client_id:>8}  {client_type:>15}  {rate:>6}  '
          f'{share:>7.4f}  {uap_sc:>10.4f}  {s3_sc:>10.4f}', flush=True)
    return {'client': client_id, 'type': client_type, 'rate': rate,
            'class0_share': share, 'uap_score': uap_sc, 's3_score': s3_sc}

print(f'\n{"client":>8}  {"type":>15}  {"rate":>6}  '
      f'{"share":>7}  {"uap_score":>10}  {"s3_score":>10}')

for c in test_adv:
    for rate in sorted(args.rates):
        rows.append(measure_client(c, 'adversarial', poison=True, rate=rate))

for c in flagged_ids:
    rows.append(measure_client(c, 'benign_flagged', poison=False, rate=float('nan')))

for c in unflagged_ids:
    rows.append(measure_client(c, 'benign_unflagged', poison=False, rate=float('nan')))

# ---------------------------------------------------------------------------
# Within-group Spearman
# ---------------------------------------------------------------------------

def group_rows(group_types):
    return [r for r in rows if r['type'] in group_types]

def sp_report(label, subset, x_key, y_key):
    x = [r[x_key] for r in subset]
    y = [r[y_key] for r in subset]
    r = spearman(x, y)
    print(f'  {label:45s}  Spearman={r:+.3f}  (n={len(subset)})')

print('\n=== Within-group Spearman(uap_score, s3_score) ===')
sp_report('adversarial only', group_rows(['adversarial']), 'uap_score', 's3_score')
sp_report('benign flagged only', group_rows(['benign_flagged']), 'uap_score', 's3_score')
sp_report('benign unflagged only', group_rows(['benign_unflagged']), 'uap_score', 's3_score')
sp_report('all benign (flagged + unflagged)', group_rows(['benign_flagged', 'benign_unflagged']), 'uap_score', 's3_score')
sp_report('all clients pooled', rows, 'uap_score', 's3_score')

print('\n=== Causal chain: Spearman(class0_share, uap_score) ===')
sp_report('adversarial only', group_rows(['adversarial']), 'class0_share', 'uap_score')
sp_report('benign unflagged only', group_rows(['benign_unflagged']), 'class0_share', 'uap_score')
sp_report('all benign', group_rows(['benign_flagged', 'benign_unflagged']), 'class0_share', 'uap_score')
sp_report('all clients pooled', rows, 'class0_share', 'uap_score')

print('\n=== Causal chain: Spearman(class0_share, s3_score) ===')
sp_report('all benign', group_rows(['benign_flagged', 'benign_unflagged']), 'class0_share', 's3_score')
sp_report('all clients pooled', rows, 'class0_share', 's3_score')

print('\n=== Group means ===')
for gtype in ['adversarial', 'benign_flagged', 'benign_unflagged']:
    g = group_rows([gtype])
    if g:
        print(f'  {gtype:20s}  share={np.mean([r["class0_share"] for r in g]):.3f}'
              f'  uap={np.mean([r["uap_score"] for r in g]):.4f}'
              f'  s3={np.mean([r["s3_score"] for r in g]):.4f}  (n={len(g)})')

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(args.out, 'check3b_uap.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['client', 'type', 'rate', 'class0_share', 'uap_score', 's3_score'])
    wr.writeheader()
    wr.writerows(rows)
print(f'\nResults saved to {csv_path}')
