#!/usr/bin/env python3
"""Check 3: UAP hypothesis test.

Compute a generic universal adversarial perturbation (UAP) toward class 0 using the
global model at the warmup checkpoint (independent of FilterFL's GAN), then measure
each client model's response to it. If Spearman(UAP_score, s3_score) is high, the
GAN probe is functionally equivalent to a UAP — the detector measures adversarial
susceptibility, not backdoor presence.

UAP computation: PGD maximising mean softmax[0] over N clean test images, using
the global (warmup) model, L-inf constraint eps.

Test clients:
  - Adversarial namelist members at --rates (default [0.3, 0.5])
  - Top-flagged benign clients from --probe-npz

    python jobs/check3_uap.py --code-dir FilterFL --out results/analysis/check3_uap \\
        --probe-npz ~/ffl_probe/s1_ours/probes.npz --skip-warmup
"""
import argparse
import copy
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--out', default='results/analysis/check3_uap')
ap.add_argument('--probe-npz', default=None)
ap.add_argument('--warmup-rounds', type=int, default=50)
ap.add_argument('--skip-warmup', action='store_true')
ap.add_argument('--warmup-dir', default=None,
                help='Directory containing warmup_ckpt.pt (default: --out)')
ap.add_argument('--rates', type=float, nargs='+', default=[0.3, 0.5],
                help='Poison rates to test for adversarial clients')
ap.add_argument('--adv-clients', type=int, nargs='+', default=None,
                help='Adversarial client IDs (default: first 4 in namelist)')
ap.add_argument('--n-benign', type=int, default=5)
ap.add_argument('--uap-eps', type=float, default=0.1,
                help='L-inf UAP budget (CIFAR-10 pixel scale 0-1)')
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
ref_main.seed(params['seed'])

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
                params=params, name=params.get('name', format(params['type'])))
handle.load_data()
handle.create_model()
print(f'adversarial_namelist: {sorted(handle.adversarial_namelist)}', flush=True)
TARGET = handle.target


# ---------------------------------------------------------------------------
# Helpers (identical to check1_prior.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# UAP computation
# ---------------------------------------------------------------------------

def compute_uap(global_model, test_loader, eps, steps, n_images):
    """PGD universal adversarial perturbation toward TARGET using the global model."""
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

    print(f'Computing UAP (eps={eps}, steps={steps}, n_images={len(images)})...',
          flush=True)
    for step in range(steps):
        opt.zero_grad()
        perturbed = (images + delta).clamp(0.0, 1.0)
        probs = torch.softmax(global_model(perturbed), dim=1)
        loss = -probs[:, TARGET].mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                p_score = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
            print(f'  step {step+1}/{steps}  global_uap_score={p_score:.4f}', flush=True)

    with torch.no_grad():
        final = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
    print(f'UAP done. Global model UAP score on {len(images)} images: {final:.4f}', flush=True)
    return delta.detach(), images


def measure_uap_score(client_model, images, uap_delta):
    """Mean softmax[TARGET] of client_model on images perturbed by uap_delta."""
    client_model.eval()
    with torch.no_grad():
        perturbed = (images + uap_delta).clamp(0.0, 1.0)
        probs = torch.softmax(client_model(perturbed), dim=1)
    return probs[:, TARGET].mean().item()


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

warmup_dir = args.warmup_dir or args.out
ckpt_path = os.path.join(warmup_dir, 'warmup_ckpt.pt')
if args.skip_warmup and os.path.exists(ckpt_path):
    print(f'Loading warm-up checkpoint from {ckpt_path}', flush=True)
    handle.model.load_state_dict(torch.load(ckpt_path, map_location=device))
elif args.warmup_rounds > 0:
    print(f'Warm-up: {args.warmup_rounds} clean FL rounds...', flush=True)
    import random as _random
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
            print(f'  round {rnd}/{args.warmup_rounds}  clean acc={acc:.1f}%', flush=True)
    params['is_poison'] = orig_is_poison
    torch.save(handle.model.state_dict(), ckpt_path)
    print(f'Checkpoint saved to {ckpt_path}', flush=True)

global_state_ref = copy.deepcopy(handle.model.state_dict())

# ---------------------------------------------------------------------------
# Compute UAP on global model
# ---------------------------------------------------------------------------

global_model = ResNet8(num_classes=10).to(device)
global_model.load_state_dict(global_state_ref)
uap_delta, uap_images = compute_uap(global_model, handle.test_data,
                                    eps=args.uap_eps, steps=args.uap_steps,
                                    n_images=args.uap_n_images)

# ---------------------------------------------------------------------------
# Test clients: adversarial at each rate, then flagged benign
# ---------------------------------------------------------------------------

adv_namelist = sorted(handle.adversarial_namelist)
test_adv = args.adv_clients if args.adv_clients else adv_namelist[:4]
test_adv = [c for c in test_adv if c in adv_namelist]

rows = []
print(f'\n{"client":>8}  {"type":>12}  {"rate":>6}  {"uap_score":>10}  {"s3_score":>10}')

for adv_client in test_adv:
    for rate in sorted(args.rates):
        params['poison_rate'] = rate
        local_state = train_client(global_state_ref, adv_client, poison=True, poison_rate=rate)
        params['poison_rate'] = params.get('poison_rate', rate)  # restore

        client_model = ResNet8(num_classes=10).to(device)
        client_model.load_state_dict(local_state)

        uap_score = measure_uap_score(client_model, uap_images, uap_delta)
        s3_score = measure_stage3(global_state_ref, local_state)

        print(f'{adv_client:>8}  {"adversarial":>12}  {rate:>6.2f}  {uap_score:>10.4f}  {s3_score:>10.4f}',
              flush=True)
        rows.append({'client': adv_client, 'type': 'adversarial', 'rate': rate,
                     'uap_score': uap_score, 's3_score': s3_score})

# Flagged benign clients
if args.probe_npz and os.path.exists(args.probe_npz):
    from collections import Counter
    d = np.load(args.probe_npz, allow_pickle=True)
    p0_all, is_adv, clients_all, std_all = d['probs'][:, 0], d['is_adv'], d['client'], d['standard']
    flagged_ben = clients_all[(~is_adv) & (p0_all > std_all)]
    top_ben = [c for c, _ in Counter(flagged_ben.tolist()).most_common(args.n_benign)]
    print(f'\nTop {len(top_ben)} flagged benign clients: {top_ben}', flush=True)

    for ben_client in top_ben:
        local_state = train_client(global_state_ref, ben_client, poison=False)
        client_model = ResNet8(num_classes=10).to(device)
        client_model.load_state_dict(local_state)

        uap_score = measure_uap_score(client_model, uap_images, uap_delta)
        s3_score = measure_stage3(global_state_ref, local_state)

        print(f'{ben_client:>8}  {"benign_flagged":>12}  {"n/a":>6}  {uap_score:>10.4f}  {s3_score:>10.4f}',
              flush=True)
        rows.append({'client': ben_client, 'type': 'benign_flagged', 'rate': float('nan'),
                     'uap_score': uap_score, 's3_score': s3_score})

# ---------------------------------------------------------------------------
# Spearman and output
# ---------------------------------------------------------------------------

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or len(set(x.tolist())) < 2:
        return float('nan')
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])

uap_scores = [r['uap_score'] for r in rows]
s3_scores  = [r['s3_score']  for r in rows]

print(f'\nSpearman(uap_score, s3_score) = {spearman(uap_scores, s3_scores):+.3f}  '
      f'(n={len(rows)}; high = GAN probe ≈ UAP)')

# Per-group breakdown
adv_rows = [r for r in rows if r['type'] == 'adversarial']
ben_rows  = [r for r in rows if r['type'] == 'benign_flagged']
if adv_rows:
    print(f'  Adversarial only  Spearman = '
          f'{spearman([r["uap_score"] for r in adv_rows], [r["s3_score"] for r in adv_rows]):+.3f}'
          f'  (n={len(adv_rows)})')
if ben_rows:
    print(f'  Benign flagged    Spearman = '
          f'{spearman([r["uap_score"] for r in ben_rows], [r["s3_score"] for r in ben_rows]):+.3f}'
          f'  (n={len(ben_rows)})')

# mean UAP score by group
if adv_rows:
    print(f'  Mean UAP score adversarial: {np.mean([r["uap_score"] for r in adv_rows]):.4f}')
if ben_rows:
    print(f'  Mean UAP score benign:      {np.mean([r["uap_score"] for r in ben_rows]):.4f}')

# Save CSV
import csv
csv_path = os.path.join(args.out, 'check3_uap.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['client', 'type', 'rate', 'uap_score', 's3_score'])
    wr.writeheader()
    wr.writerows(rows)
print(f'\nResults saved to {csv_path}')
