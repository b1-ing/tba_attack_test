#!/usr/bin/env python3
"""Check 10: What per-update property does Stage-3 actually read?

For each (client, rate, seed), trains one adversarial round from the warmup
checkpoint and measures five weight-space quantities alongside Stage-3.
The goal is to find what scales monotonically with rate across all clients.

Clients: 34, 36, 45, 47 (check3b reference set)
Rates: 0.10, 0.20, 0.25, 0.30, 0.40, 0.50
Seeds: 1, 2, 3

Output: raw table first, Spearman correlations second.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out', default='.')
ap.add_argument('--clients', type=int, nargs='+', default=[34, 36, 45, 47])
ap.add_argument('--rates', type=float, nargs='+',
                default=[0.10, 0.20, 0.25, 0.30, 0.40, 0.50])
ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--alpha', type=float, default=0.5)
ap.add_argument('--global-seed', type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import yaml

import main as ref_main
import train as train_mod
from gan import gan_defense
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['seed'] = args.global_seed
params['dirichlet_alpha'] = args.alpha
params['defence_method'] = 'ours'
ref_main.seed(params['seed'])

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET = handle.target  # class 0

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'No warmup checkpoint at {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

global_state = copy.deepcopy(handle.model.state_dict())
NUM_CLASSES = 10


def features(model, x):
    """Forward pass up to (but not including) the final linear layer."""
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def measure(client_id, rate, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate'] = rate
    p['uap_mu'] = 0.0

    model = ResNet8(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(copy.deepcopy(global_state))

    w_local = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id,
        params=p,
        model=model,
        dataset=handle.train_data[client_id],
        handle=handle,
        poison=True,
    )

    # ── Weight-space quantities ──────────────────────────────────────────────
    W_g = global_state['linear1.weight']          # (10, 512)
    b_g = global_state['linear1.bias']            # (10,)
    W_l = w_local['linear1.weight']
    b_l = w_local['linear1.bias']

    # 1. ||W_fc[target]|| in local model
    fc_norm_local  = W_l[TARGET].norm().item()
    fc_norm_global = W_g[TARGET].norm().item()

    # 2. Relative norm: target row vs mean of all rows (local model)
    row_norms = W_l.norm(dim=1)                   # (10,)
    fc_rel_norm = (row_norms[TARGET] / row_norms.mean()).item()

    # 3. Target-class bias (local)
    fc_bias = b_l[TARGET].item()
    fc_bias_delta = (b_l[TARGET] - b_g[TARGET]).item()

    # 4. ||w_local - w_global|| overall (Frobenius / param count)
    delta_norms = {}
    total_sq, total_n = 0.0, 0
    for k in global_state:
        if global_state[k].dtype == torch.int64:
            continue
        d = (w_local[k].float() - global_state[k].float())
        delta_norms[k] = d.norm().item()
        total_sq += d.pow(2).sum().item()
        total_n   += d.numel()
    update_rms = (total_sq / total_n) ** 0.5

    # 5. Delta in final-layer target row vs delta for other rows
    delta_fc_target = (W_l[TARGET] - W_g[TARGET])
    delta_fc_others = torch.stack([W_l[i] - W_g[i]
                                   for i in range(NUM_CLASSES) if i != TARGET])
    fc_delta_target_norm = delta_fc_target.norm().item()
    fc_delta_others_norm = delta_fc_others.norm(dim=1).mean().item()
    fc_delta_ratio = fc_delta_target_norm / (fc_delta_others_norm + 1e-9)

    # 6. Cosine similarity: direction of delta_fc_target vs W_l[target]
    cos = torch.nn.functional.cosine_similarity(
        delta_fc_target.unsqueeze(0), W_l[TARGET].unsqueeze(0)).item()

    # ── Stage-3 score ────────────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    g_imgs = gan_defense(global_state, w_local, params, the_epoch=1)
    probe_imgs = g_imgs[0]  # (N, C, H, W)

    m = ResNet8(num_classes=NUM_CLASSES).to(device)
    m.load_state_dict(w_local)
    m.eval()
    with torch.no_grad():
        logits = m(probe_imgs)                    # (N, 10)
        probs  = torch.softmax(logits, dim=1)
        s3     = probs.mean(0)[TARGET].item()
        mean_target_logit = logits[:, TARGET].mean().item()
        feats  = features(m, probe_imgs)          # (N, 512)

    # Mean feature norm (how strongly the probe activates the penultimate layer)
    mean_feat_norm = feats.norm(dim=1).mean().item()

    # Alignment of mean probe feature with W_l[target]
    mean_feat = feats.mean(0)  # (512,)
    feat_align = torch.nn.functional.cosine_similarity(
        mean_feat.unsqueeze(0), W_l[TARGET].unsqueeze(0)).item()

    return {
        'client': client_id, 'rate': rate, 'seed': seed,
        'fc_norm_local':       fc_norm_local,
        'fc_norm_global':      fc_norm_global,
        'fc_norm_delta':       fc_norm_local - fc_norm_global,
        'fc_rel_norm':         fc_rel_norm,
        'fc_bias':             fc_bias,
        'fc_bias_delta':       fc_bias_delta,
        'update_rms':          update_rms,
        'fc_delta_target_norm': fc_delta_target_norm,
        'fc_delta_others_norm': fc_delta_others_norm,
        'fc_delta_ratio':       fc_delta_ratio,
        'cos_delta_vs_weight':  cos,
        'mean_target_logit':   mean_target_logit,
        'mean_feat_norm':      mean_feat_norm,
        'feat_align':          feat_align,
        's3':                  s3,
    }


COLS = ['client', 'rate', 'seed',
        'fc_norm_local', 'fc_norm_delta', 'fc_rel_norm',
        'fc_bias_delta', 'update_rms',
        'fc_delta_target_norm', 'fc_delta_ratio',
        'cos_delta_vs_weight',
        'mean_target_logit', 'feat_align', 's3']

rows = []
print(f"\n{'client':>8}  {'rate':>5}  {'seed':>4}  "
      f"{'fc_Δnorm':>9}  {'fc_rel':>7}  {'bias_Δ':>7}  "
      f"{'upd_rms':>8}  {'fcΔ_ratio':>10}  {'cos_Δ,W':>8}  "
      f"{'logit_t':>8}  {'feat_al':>8}  {'s3':>7}", flush=True)
print('-' * 110)

for client_id in args.clients:
    for rate in args.rates:
        for seed in args.seeds:
            r = measure(client_id, rate, seed)
            rows.append(r)
            print(f"{r['client']:>8}  {r['rate']:>5.2f}  {r['seed']:>4}  "
                  f"{r['fc_norm_delta']:>9.4f}  {r['fc_rel_norm']:>7.4f}  "
                  f"{r['fc_bias_delta']:>7.4f}  {r['update_rms']:>8.5f}  "
                  f"{r['fc_delta_ratio']:>10.4f}  {r['cos_delta_vs_weight']:>8.4f}  "
                  f"{r['mean_target_logit']:>8.4f}  {r['feat_align']:>8.4f}  "
                  f"{r['s3']:>7.4f}", flush=True)

# ── Per-client Spearman: rate → each quantity ─────────────────────────────
def spearman(x, y):
    n = len(x)
    if n < 3: return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d  = rx - ry
    return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))

quantities = ['fc_norm_delta', 'fc_rel_norm', 'fc_bias_delta', 'update_rms',
              'fc_delta_target_norm', 'fc_delta_ratio', 'cos_delta_vs_weight',
              'mean_target_logit', 'feat_align']

print('\n\n=== Spearman(rate → quantity) and Spearman(quantity → s3) ===')
print('Reported per-client (use pooled only as sanity check — n=6 rates × 3 seeds)')
print()
for q in quantities:
    print(f'\n  {q}')
    print(f"  {'client':>8}  {'r(rate,q)':>10}  {'r(q,s3)':>9}")
    pooled_rate, pooled_q, pooled_s3 = [], [], []
    for c in args.clients:
        cr = [r for r in rows if r['client'] == c]
        rates_ = [r['rate'] for r in cr]
        qvals  = [r[q]     for r in cr]
        s3vals = [r['s3']  for r in cr]
        r_rq = spearman(rates_, qvals)
        r_qs = spearman(qvals, s3vals)
        print(f"  {c:>8}  {r_rq:>10.3f}  {r_qs:>9.3f}")
        pooled_rate += rates_; pooled_q += qvals; pooled_s3 += s3vals
    r_pool_rq = spearman(pooled_rate, pooled_q)
    r_pool_qs = spearman(pooled_q, pooled_s3)
    print(f"  {'pooled':>8}  {r_pool_rq:>10.3f}  {r_pool_qs:>9.3f}")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out, 'check10_weight_analysis.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
print(f'\nSaved to {csv_path}')
