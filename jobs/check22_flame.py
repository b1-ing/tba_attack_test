#!/usr/bin/env python3
"""Check 22: FLAME defense against cov_64_pos backdoor attack.

FLAME (USENIX Security 2022) aggregates in three steps each round:
  1. Cluster: pairwise cosine distances on flattened model updates,
     HDBSCAN(min_cluster_size = floor(n/2)+1) to find the majority cluster;
     exclude clients outside the largest cluster.
  2. Clip: clip each kept update to the median L2 norm S_t of the cluster.
  3. Noise: add Gaussian noise N(0, (λ × S_t)²) per parameter to the
     aggregated update. λ is the noise multiplier (the key hyperparameter).

The clustering step catches clients with dramatically different updates
(e.g., model-replacement attacks). Our attack (cov_64_pos at 50% poison
rate) mostly does clean training, so adversarial clients likely cluster
with benign clients. The noise step is what would prevent the backdoor
from accumulating even if clustering misses the attack.

This check sweeps λ ∈ {0.000, 0.001, 0.010, 0.100} plus a FedAvg baseline.
  λ=0.000: clustering + clipping only (no noise) — tests if HDBSCAN catches it
  λ=0.001: paper's conservative noise setting
  λ=0.010: moderate noise
  λ=0.100: aggressive noise (may degrade clean accuracy significantly)

For each condition we record: excluded clients per round, S_t, σ, ASR on
global model, clean accuracy.

5 conditions × 25 rounds × 10 clients = 1250 client-round measurements. ~3h.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',      default='FilterFL')
ap.add_argument('--warmup-dir',    default=None)
ap.add_argument('--out',           default='.')
ap.add_argument('--adv-clients',   type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--ben-clients',   type=int, nargs='+', default=[0, 1, 2, 3, 5, 10])
ap.add_argument('--n-rounds',      type=int, default=25)
ap.add_argument('--lambda-vals',   type=float, nargs='+',
                default=[0.000, 0.001, 0.010, 0.100])
ap.add_argument('--poison-rate',   type=float, default=0.50)
ap.add_argument('--amplitude',     type=float, default=0.5)
ap.add_argument('--n-test',        type=int, default=500)
ap.add_argument('--alpha',         type=float, default=0.5)
ap.add_argument('--global-seed',   type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.nn.functional as F
import yaml
from hdbscan import HDBSCAN

import main as ref_main
import train as train_mod
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
TARGET = handle.target
C, H, W = 3, 32, 32
print(f'Target class: {TARGET}', flush=True)

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'Warmup checkpoint not found: {ckpt}'

# ── Test pool ─────────────────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs).to(device)
all_lbls = torch.cat(all_lbls).to(device)
non_target = all_imgs[all_lbls != TARGET][:args.n_test]
target_imgs = all_imgs[all_lbls == TARGET][:args.n_test]

# ── Trigger (cov_64_pos) ─────────────────────────────────────────────────────
rng64 = np.random.RandomState(42)
pos64 = rng64.choice(H * W, size=64, replace=False)
trigger_np = np.zeros((C, H, W), dtype=np.float32)
for p in pos64:
    h_p, w_p = divmod(int(p), W)
    trigger_np[:, h_p, w_p] = args.amplitude
trigger_t = torch.from_numpy(trigger_np).to(device)

# ── Flatten helper ────────────────────────────────────────────────────────────
def flatten(state_dict):
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for v in state_dict.values()])

def unflatten_into(agg_flat, ref_state):
    new_state = {}
    offset = 0
    for k, v in ref_state.items():
        numel = v.numel()
        chunk = agg_flat[offset:offset + numel].reshape(v.shape)
        new_state[k] = v.cpu().float() + torch.from_numpy(chunk.astype(np.float32))
        offset += numel
    return new_state

# ── FLAME aggregation ─────────────────────────────────────────────────────────
def flame_aggregate(global_state, local_states, client_ids, lambda_noise, rng):
    n = len(local_states)
    g_flat = flatten(global_state)

    deltas = np.stack([flatten(w) - g_flat for w in local_states])  # (n, D)

    # Cosine distance matrix
    norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-9
    normed = deltas / norms
    cos_sim = normed @ normed.T
    cos_dist = np.clip(1.0 - cos_sim, 0.0, 2.0).astype(np.float64)

    # HDBSCAN majority clustering
    min_cs = max(2, n // 2 + 1)
    try:
        labels = HDBSCAN(min_cluster_size=min_cs, metric='precomputed').fit_predict(cos_dist)
    except Exception:
        labels = np.zeros(n, dtype=int)   # fallback: keep all

    if labels.max() >= 0:
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest = unique[counts.argmax()]
        keep = labels == largest
    else:
        keep = np.ones(n, dtype=bool)

    kept_deltas = deltas[keep]
    excluded_ids = [client_ids[i] for i in range(n) if not keep[i]]
    kept_ids     = [client_ids[i] for i in range(n) if keep[i]]

    # Clip to median L2 norm
    kept_norms = np.linalg.norm(kept_deltas, axis=1)
    S_t = float(np.median(kept_norms))
    clipped = np.stack([
        d / max(1.0, float(np.linalg.norm(d)) / (S_t + 1e-9))
        for d in kept_deltas
    ])

    # Average + Gaussian noise
    agg = clipped.mean(axis=0)
    sigma = lambda_noise * S_t
    if sigma > 0:
        agg = agg + rng.normal(0.0, sigma, agg.shape).astype(np.float32)

    new_state = unflatten_into(agg, global_state)
    return new_state, kept_ids, excluded_ids, S_t, sigma

# ── FedAvg aggregation ─────────────────────────────────────────────────────────
def fedavg_aggregate(global_state, local_states, client_ids, sizes):
    total = sum(sizes)
    g_flat = flatten(global_state)
    deltas = np.stack([flatten(w) - g_flat for w in local_states])
    weights = np.array(sizes, dtype=np.float32) / total
    agg = (deltas * weights[:, None]).sum(axis=0)
    return unflatten_into(agg, global_state)

# ── Metric: ASR + clean acc on global model ───────────────────────────────────
def global_metrics(global_state):
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    m.eval()
    with torch.no_grad():
        trig_imgs = torch.clamp(non_target + trigger_t.unsqueeze(0), -1., 1.)
        asr = torch.softmax(m(trig_imgs), 1).mean(0)[TARGET].item()
        preds = torch.softmax(m(non_target), 1).argmax(1)
        clean_acc = (preds != TARGET).float().mean().item()
    return asr, clean_acc

# ── Client training ────────────────────────────────────────────────────────────
ADV_SET = set(args.adv_clients)

def train_client(global_state, client_id, round_seed):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    p = copy.deepcopy(params)
    p['poison_rate'] = args.poison_rate
    import train as tm
    _orig = tm.apply_pattern_batch_gpu
    is_adv = client_id in ADV_SET
    if is_adv:
        _pt = trigger_t.cpu()
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat = _pt.to(images.device)
            poisoned = torch.clamp(images + pat.unsqueeze(0), -1.0, 1.0)
            labels = torch.full((images.size(0),), TARGET,
                                device=images.device, dtype=torch.long)
            return (torch.where(pm_exp.expand_as(images).bool(), poisoned, images),
                    labels)
        tm.apply_pattern_batch_gpu = patched
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    try:
        w_local = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id, params=p, model=m,
            dataset=handle.train_data[client_id], handle=handle,
            poison=is_adv,
        )
    finally:
        if is_adv:
            tm.apply_pattern_batch_gpu = _orig
    return {k: v.cpu() for k, v in w_local.items()}

# ── Main loop ─────────────────────────────────────────────────────────────────
ALL_CLIENTS = args.adv_clients + args.ben_clients
CONDITIONS = [('fedavg', None)] + [(f'flame_l{lv:.3f}', lv) for lv in args.lambda_vals]

COLS = ['condition', 'round', 'client', 'role',
        'excluded', 'S_t', 'sigma', 'asr_global', 'clean_acc']
all_rows = []

for cond_key, lambda_val in CONDITIONS:
    is_flame = lambda_val is not None
    lbl = f'FedAvg' if not is_flame else f'FLAME λ={lambda_val:.3f}'
    print(f'\n{"="*70}\nCondition: {lbl}\n{"="*70}', flush=True)

    handle.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
    rng_noise = np.random.RandomState(args.global_seed)

    excl_counts = {cid: 0 for cid in ALL_CLIENTS}

    print(f'\n{"rnd":>4}  {"excl":>20}  {"S_t":>8}  {"sigma":>8}  '
          f'{"ASR":>7}  {"clean":>7}')
    print('-' * 62)

    for rnd in range(1, args.n_rounds + 1):
        local_states, sizes = [], []
        for cid in ALL_CLIENTS:
            w_local = train_client(global_state, cid, round_seed=rnd * 1000 + cid)
            local_states.append(w_local)
            sizes.append(handle.clients_data_num[cid])

        if is_flame:
            global_state, kept_ids, excl_ids, S_t, sigma = flame_aggregate(
                global_state, local_states, ALL_CLIENTS, lambda_val, rng_noise)
            for cid in excl_ids:
                excl_counts[cid] += 1
        else:
            global_state = fedavg_aggregate(global_state, local_states,
                                            ALL_CLIENTS, sizes)
            excl_ids, S_t, sigma = [], float('nan'), float('nan')

        asr, clean_acc = global_metrics(global_state)

        excl_str = str(excl_ids) if excl_ids else '[]'
        print(f'{rnd:>4}  {excl_str:>20}  {S_t:>8.2f}  {sigma:>8.4f}  '
              f'{asr:>7.4f}  {clean_acc:>7.4f}', flush=True)

        for i, cid in enumerate(ALL_CLIENTS):
            role = 'ADV' if cid in ADV_SET else 'BEN'
            excluded = cid in excl_ids
            all_rows.append({
                'condition': cond_key, 'round': rnd, 'client': cid,
                'role': role, 'excluded': int(excluded),
                'S_t': S_t, 'sigma': sigma,
                'asr_global': asr, 'clean_acc': clean_acc,
            })

    # Per-condition summary
    print(f'\n=== Exclusion counts (out of {args.n_rounds} rounds) ===')
    for cid in ALL_CLIENTS:
        role = 'ADV' if cid in ADV_SET else 'BEN'
        print(f'  client {cid:>3} ({role}): excluded {excl_counts[cid]:>3}x')
    asr_f, clean_f = global_metrics(global_state)
    print(f'  Final ASR={asr_f:.4f}  clean_acc={clean_f:.4f}')

# ── Cross-condition summary ────────────────────────────────────────────────────
print('\n\n=== Cross-condition summary (round 25 values) ===')
print(f'{"condition":>20}  {"ASR":>7}  {"clean":>7}  '
      f'{"ADV excl%":>10}  {"BEN excl%":>10}  {"σ_mean":>8}')
print('-' * 70)
for cond_key, _ in CONDITIONS:
    rnd_rows = [r for r in all_rows if r['condition'] == cond_key]
    asr_vals = [float(r['asr_global']) for r in rnd_rows
                if r['client'] == ALL_CLIENTS[0]]
    adv_excl = [r['excluded'] for r in rnd_rows if r['role'] == 'ADV']
    ben_excl = [r['excluded'] for r in rnd_rows if r['role'] == 'BEN']
    sig_vals  = [float(r['sigma']) for r in rnd_rows
                 if r['client'] == ALL_CLIENTS[0] and r['sigma'] != float('nan')]
    last_asr   = asr_vals[-1] if asr_vals else float('nan')
    last_clean = float([r['clean_acc'] for r in rnd_rows
                        if r['client'] == ALL_CLIENTS[0]][-1])
    adv_pct = 100 * np.mean(adv_excl) if adv_excl else float('nan')
    ben_pct = 100 * np.mean(ben_excl) if ben_excl else float('nan')
    sig_m   = np.mean([s for s in sig_vals if not np.isnan(s)]) if sig_vals else float('nan')
    print(f'{cond_key:>20}  {last_asr:>7.4f}  {last_clean:>7.4f}  '
          f'{adv_pct:>10.1f}  {ben_pct:>10.1f}  {sig_m:>8.4f}')

csv_path = os.path.join(args.out, 'check22_flame.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(all_rows)
print(f'\nSaved to {csv_path}')
