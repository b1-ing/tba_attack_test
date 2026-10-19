#!/usr/bin/env python3
"""Check 4: Single-client UAP-suppression loss sweep.

Trains one adversarial client at several uap_mu values and reports:
  - local ASR (does the backdoor still work?)
  - UAP score (does susceptibility to a class-0 UAP drop?)
  - Stage-3 score (does FilterFL's probe score drop?)

Use client 45 at rate=0.5 by default: highest UAP/s3 in check3b, most
detectable, best signal for whether the suppression term has any effect.

    python jobs/check4_uap_suppress.py --code-dir FilterFL --out results/analysis/check4 \
        --warmup-dir ~/ffl_check1 --skip-warmup
"""
import argparse
import copy
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--out', default='results/analysis/check4')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--skip-warmup', action='store_true')
ap.add_argument('--warmup-rounds', type=int, default=50)
ap.add_argument('--adv-client', type=int, default=None)
ap.add_argument('--adv-clients', type=int, nargs='+', default=None)
ap.add_argument('--rate', type=float, default=0.5)
ap.add_argument('--mu-values', type=float, nargs='+', default=[0.0, 0.01, 0.1, 0.5])
ap.add_argument('--uap-eps', type=float, default=0.1)
ap.add_argument('--uap-steps', type=int, default=300)   # for measuring UAP score
ap.add_argument('--uap-steps-train', type=int, default=30)  # for the suppression term
ap.add_argument('--uap-n-images', type=int, default=512)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--local-seeds', type=int, nargs='+', default=[1],
                help='Seeds for local training randomness; repeat each cell once per seed')
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

# Resolve client list: --adv-clients > --adv-client > all 4
if args.adv_clients:
    test_clients = [c for c in args.adv_clients if c in adv_namelist]
elif args.adv_client is not None:
    test_clients = [args.adv_client] if args.adv_client in adv_namelist else []
else:
    test_clients = adv_namelist[:4]
assert test_clients, f'No valid adversarial clients in {adv_namelist}'
print(f'Test clients: {test_clients}', flush=True)


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
# Helpers
# ---------------------------------------------------------------------------

def compute_uap(global_model, test_loader, eps, steps, n_images):
    global_model.eval()
    batches, collected = [], 0
    for data, _ in test_loader:
        batches.append(data)
        collected += len(data)
        if collected >= n_images:
            break
    images = torch.cat(batches)[:n_images].to(device)
    delta = torch.zeros(1, *images.shape[1:], device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.01)
    print(f'Computing measurement UAP (eps={eps}, steps={steps}, n={len(images)})...', flush=True)
    for step in range(steps):
        opt.zero_grad()
        (-torch.softmax(global_model((images + delta).clamp(0, 1)), dim=1)[:, TARGET].mean()).backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                s = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
            print(f'  step {step+1}  global_uap_score={s:.4f}', flush=True)
    with torch.no_grad():
        s = torch.softmax(global_model((images + delta).clamp(0, 1)), 1)[:, TARGET].mean()
    print(f'UAP done. Global model score: {s:.4f}', flush=True)
    return delta.detach(), images


def measure_uap_score(client_model, images, uap_delta):
    client_model.eval()
    with torch.no_grad():
        probs = torch.softmax(client_model((images + uap_delta).clamp(0, 1)), dim=1)
    return probs[:, TARGET].mean().item()


def measure_stage3(global_state, local_state):
    g_imgs = gan_defense(global_state, local_state, params, the_epoch=1)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    m.eval()
    with torch.no_grad():
        out = torch.softmax(m(g_imgs[0]), dim=1)
    return out.mean(0)[TARGET].item()


def measure_asr(local_state):
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    asr = test_mod.normal_test(m, handle.test_un_target_label_data, params, handle, poison=True)
    return asr


# ---------------------------------------------------------------------------
# Pre-compute measurement UAP (once, from global model)
# ---------------------------------------------------------------------------

global_model = ResNet8(num_classes=10).to(device)
global_model.load_state_dict(global_state_ref)
uap_delta, uap_images = compute_uap(global_model, handle.test_data,
                                    eps=args.uap_eps, steps=args.uap_steps,
                                    n_images=args.uap_n_images)


# ---------------------------------------------------------------------------
# Sweep over clients × uap_mu values
# ---------------------------------------------------------------------------

print(f'\n{"client":>8}  {"mu":>6}  {"seed":>5}  {"uap_score":>10}  {"s3_score":>10}  {"asr":>8}', flush=True)
rows = []

for client_id in test_clients:
    for mu in args.mu_values:
        for local_seed in args.local_seeds:
            torch.manual_seed(local_seed)
            np.random.seed(local_seed)

            p = copy.deepcopy(params)
            p['poison_rate'] = args.rate
            p['uap_mu'] = mu
            p['uap_eps'] = args.uap_eps
            p['uap_steps_train'] = args.uap_steps_train
            p['uap_n_images_train'] = 128

            model = ResNet8(num_classes=10).to(device)
            model.load_state_dict(copy.deepcopy(global_state_ref))

            local_state = train_mod.standard_train(
                epoch=1,
                dataset_size=handle.clients_data_num[client_id],
                client=client_id,
                params=p,
                model=model,
                dataset=handle.train_data[client_id],
                handle=handle,
                poison=True,
            )

            client_model = ResNet8(num_classes=10).to(device)
            client_model.load_state_dict(local_state)

            uap_sc = measure_uap_score(client_model, uap_images, uap_delta)
            s3_sc  = measure_stage3(global_state_ref, local_state)
            asr    = measure_asr(local_state)

            print(f'{client_id:>8}  {mu:>6.3f}  {local_seed:>5}  '
                  f'{uap_sc:>10.4f}  {s3_sc:>10.4f}  {asr:>8.2f}%', flush=True)
            rows.append({'client': client_id, 'mu': mu, 'seed': local_seed,
                         'uap_score': uap_sc, 's3_score': s3_sc, 'asr': asr})

# Per-cell mean ± std summary
print('\n=== Mean ± std across seeds ===')
print(f'{"client":>8}  {"mu":>6}  {"uap_mean":>9}  {"uap_std":>8}  '
      f'{"s3_mean":>8}  {"s3_std":>7}  {"asr_mean":>9}  {"asr_std":>8}')
cells = {}
for r in rows:
    k = (r['client'], r['mu'])
    cells.setdefault(k, []).append(r)
for (client_id, mu), cell_rows in sorted(cells.items()):
    uap_vals = [r['uap_score'] for r in cell_rows]
    s3_vals  = [r['s3_score']  for r in cell_rows]
    asr_vals = [r['asr']       for r in cell_rows]
    print(f'{client_id:>8}  {mu:>6.3f}  '
          f'{np.mean(uap_vals):>9.4f}  {np.std(uap_vals):>8.4f}  '
          f'{np.mean(s3_vals):>8.4f}  {np.std(s3_vals):>7.4f}  '
          f'{np.mean(asr_vals):>9.2f}%  {np.std(asr_vals):>7.2f}%')

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(args.out, 'check4_uap_suppress.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['client', 'mu', 'seed', 'uap_score', 's3_score', 'asr'])
    wr.writeheader()
    wr.writerows(rows)
print(f'\nResults saved to {csv_path}')
