#!/usr/bin/env python3
"""Check 5: Stage-3 variance decomposition.

Separates Stage-3 score variance into two axes:
  - Attacker training seed  (local SGD trajectory, poison mask, data order)
  - GAN probe seed          (Stage-3 GAN training randomness)

For each client, trains the adversarial model once per attacker seed, then
evaluates Stage-3 once per GAN seed — giving a full attacker×GAN factorial.

Columns of interest:
  s3_var_attacker  — std across attacker seeds with GAN seed fixed
  s3_var_gan       — std across GAN seeds with attacker seed fixed

If the GAN seed drives the spread, probe noise is the culprit.
If attacker seed drives it, the local model is genuinely variable.
"""
import argparse
import copy
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--out', default='results/analysis/check5')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--skip-warmup', action='store_true')
ap.add_argument('--warmup-rounds', type=int, default=50)
ap.add_argument('--clients', type=int, nargs='+', default=[45, 47])
ap.add_argument('--rate', type=float, default=0.5)
ap.add_argument('--attacker-seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
ap.add_argument('--gan-seeds',      type=int, nargs='+', default=[1, 2, 3, 4, 5])
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=1)
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

test_clients = [c for c in args.clients if c in adv_namelist]
assert test_clients, f'No valid adversarial clients in {adv_namelist}'
print(f'Test clients: {test_clients}', flush=True)
print(f'Attacker seeds: {args.attacker_seeds}', flush=True)
print(f'GAN seeds: {args.gan_seeds}', flush=True)

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
        w_locals = []
        for c in clients:
            p = copy.deepcopy(params)
            m = ResNet8(num_classes=10).to(device)
            m.load_state_dict(copy.deepcopy(ori_w))
            w = train_mod.standard_train(1, handle.clients_data_num[c], c, p, m,
                                         handle.train_data[c], handle, poison=False)
            w_locals.append(w)
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
# Seeded Stage-3 measurement
# ---------------------------------------------------------------------------

def measure_stage3_seeded(global_state, local_state, gan_seed):
    """Run Stage-3 GAN probe with a fixed seed so the GAN axis is controlled."""
    torch.manual_seed(gan_seed)
    np.random.seed(gan_seed)
    g_imgs = gan_defense(global_state, local_state, params, the_epoch=1)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    m.eval()
    with torch.no_grad():
        out = torch.softmax(m(g_imgs[0]), dim=1)
    return out.mean(0)[TARGET].item()


# ---------------------------------------------------------------------------
# Full factorial sweep: attacker_seed × gan_seed
# ---------------------------------------------------------------------------

rows = []

for client_id in test_clients:
    print(f'\n--- Client {client_id} ---', flush=True)

    # Step 1: train one local model per attacker seed
    local_states = {}
    for a_seed in args.attacker_seeds:
        torch.manual_seed(a_seed)
        np.random.seed(a_seed)

        p = copy.deepcopy(params)
        p['poison_rate'] = args.rate
        p['uap_mu'] = 0.0  # no suppression — pure variance measurement

        model = ResNet8(num_classes=10).to(device)
        model.load_state_dict(copy.deepcopy(global_state_ref))

        local_states[a_seed] = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id,
            params=p,
            model=model,
            dataset=handle.train_data[client_id],
            handle=handle,
            poison=True,
        )
        print(f'  trained attacker_seed={a_seed}', flush=True)

    # Step 2: evaluate Stage-3 for each (attacker_seed, gan_seed) pair
    print(f'\n{"a_seed":>8}  {"g_seed":>7}  {"s3":>8}', flush=True)
    for a_seed in args.attacker_seeds:
        for g_seed in args.gan_seeds:
            s3 = measure_stage3_seeded(global_state_ref, local_states[a_seed], g_seed)
            print(f'{client_id:>4}  {a_seed:>8}  {g_seed:>7}  {s3:>8.4f}', flush=True)
            rows.append({'client': client_id, 'attacker_seed': a_seed,
                         'gan_seed': g_seed, 's3': s3})

# ---------------------------------------------------------------------------
# Variance decomposition summary
# ---------------------------------------------------------------------------

print('\n=== Variance decomposition ===')
print(f'{"client":>8}  {"axis":>14}  {"fixed_seed":>11}  {"s3_mean":>8}  {"s3_std":>7}')

for client_id in test_clients:
    client_rows = [r for r in rows if r['client'] == client_id]

    # Vary GAN seed, fix attacker seed → GAN noise
    for a_seed in args.attacker_seeds:
        vals = [r['s3'] for r in client_rows if r['attacker_seed'] == a_seed]
        print(f'{client_id:>8}  {"vary_gan":>14}  {a_seed:>11}  '
              f'{np.mean(vals):>8.4f}  {np.std(vals):>7.4f}')

    # Vary attacker seed, fix GAN seed → training noise
    for g_seed in args.gan_seeds:
        vals = [r['s3'] for r in client_rows if r['gan_seed'] == g_seed]
        print(f'{client_id:>8}  {"vary_attacker":>14}  {g_seed:>11}  '
              f'{np.mean(vals):>8.4f}  {np.std(vals):>7.4f}')

# Aggregate: mean std for each axis
print('\n=== Aggregate std per axis ===')
for client_id in test_clients:
    client_rows = [r for r in rows if r['client'] == client_id]
    gan_stds = []
    for a_seed in args.attacker_seeds:
        vals = [r['s3'] for r in client_rows if r['attacker_seed'] == a_seed]
        gan_stds.append(np.std(vals))
    att_stds = []
    for g_seed in args.gan_seeds:
        vals = [r['s3'] for r in client_rows if r['gan_seed'] == g_seed]
        att_stds.append(np.std(vals))
    print(f'client {client_id}:  '
          f'GAN-axis std = {np.mean(gan_stds):.4f} ± {np.std(gan_stds):.4f}  |  '
          f'Attacker-axis std = {np.mean(att_stds):.4f} ± {np.std(att_stds):.4f}')

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(args.out, 'check5_s3_variance.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['client', 'attacker_seed', 'gan_seed', 's3'])
    wr.writeheader()
    wr.writerows(rows)
print(f'\nResults saved to {csv_path}')
