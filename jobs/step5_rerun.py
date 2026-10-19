#!/usr/bin/env python3
"""Step 5: Full 600-round defense comparison at check12b setup.

Fixes applied vs checks 19-22:
  FLTrust:   params-only flatten (no BN buffers), local_epochs key fixed,
             logs ||Δ_params||₂ and ||Δ_buffers||₂ separately (Step 2b),
             measures trust score for ALL selected clients (ADV and BEN).
  FoolsGold: params-only flatten (no BN buffers).
  FLAME:     allow_single_cluster=True (fixes all-noise fallback).
  All:       true clean_acc (correct-class fraction on full test set),
             non_target_tnr (fraction non-target not classified as target).

Setup (check12b):
  100 clients, 10 selected/round, 10 ADV, α=0.5, poison_rate=0.5,
  local_epochs=2, lr=0.01, momentum=0.9, seed=1, 600 rounds.

Args: --defense {fedavg,fltrust,foolsgold,flame}
      --trigger {checker,cov64pos,patch}
      --alpha FLOAT (default 0.5; use 100 for near-IID control)
"""
import argparse, copy, csv, datetime, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--defense',       required=True,
                choices=['fedavg', 'fltrust', 'foolsgold', 'flame'])
ap.add_argument('--trigger',       required=True,
                choices=['checker', 'cov64pos', 'patch'])
ap.add_argument('--code-dir',      default='FilterFL')
ap.add_argument('--warmup-dir',    default=None)
ap.add_argument('--out',           default='.')
ap.add_argument('--n-rounds',      type=int,   default=600)
ap.add_argument('--n-clients',     type=int,   default=10)
ap.add_argument('--n-total',       type=int,   default=100)
ap.add_argument('--n-adv',         type=int,   default=10)
ap.add_argument('--alpha',         type=float, default=0.5)
ap.add_argument('--seed',          type=int,   default=1)
ap.add_argument('--poison-rate',   type=float, default=0.5)
ap.add_argument('--local-epochs',  type=int,   default=2)
ap.add_argument('--lr',            type=float, default=0.01)
ap.add_argument('--momentum',      type=float, default=0.9)
ap.add_argument('--amplitude',     type=float, default=0.5)
ap.add_argument('--root-per-class',type=int,   default=10)
ap.add_argument('--lambda-flame',  type=float, default=0.001,
                help='FLAME noise multiplier')
ap.add_argument('--checkpoint-every', type=int, default=50)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
from hdbscan import HDBSCAN

import main as ref_main
import train as train_mod
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params['seed']                    = args.seed
params['dirichlet_alpha']         = args.alpha
params['defence_method']          = 'ours'
params['number_of_total_participants'] = args.n_total
params['num_clients']             = args.n_clients
params['poison_client_num']       = args.n_adv
params['local_epochs']            = args.local_epochs
params['lr']                      = args.lr
params['momentum']                = args.momentum
params['poison_rate']             = args.poison_rate
params['is_poison']               = True
ref_main.seed(args.seed)

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET = handle.target
N_CLASSES = 10
C, H, W = 3, 32, 32
print(f'Target class: {TARGET}', flush=True)
print(f'Defense: {args.defense}  Trigger: {args.trigger}  α={args.alpha}', flush=True)
print(f'ADV clients: {handle.adversarial_namelist[:5]}... ({len(handle.adversarial_namelist)} total)',
      flush=True)

# ── Warmup checkpoint ─────────────────────────────────────────────────────────
warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'Warmup checkpoint not found: {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

# ── Test images ───────────────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs)   # CPU
all_lbls = torch.cat(all_lbls)   # CPU
n_test_total = len(all_imgs)

all_imgs_gpu = all_imgs.to(device)
all_lbls_gpu = all_lbls.to(device)
non_target_gpu = all_imgs_gpu[all_lbls_gpu != TARGET]

# ── Trigger patterns ──────────────────────────────────────────────────────────
def _make_trigger():
    if args.trigger == 'cov64pos':
        rng = np.random.RandomState(42)
        pos = rng.choice(H * W, size=64, replace=False)
        pat = np.zeros((C, H, W), dtype=np.float32)
        for p in pos:
            h_p, w_p = divmod(int(p), W)
            pat[:, h_p, w_p] = args.amplitude
        return 'additive', pat
    elif args.trigger == 'checker':
        rows_c = np.arange(H).reshape(H, 1)
        cols_c = np.arange(W).reshape(1, W)
        hw = ((rows_c + cols_c) % 2).astype(np.float32) * 2.0 - 1.0
        pat = np.stack([hw * args.amplitude] * C, axis=0)
        return 'additive', pat
    else:  # patch
        _cpu = torch.device('cpu')
        mask_t = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)
        return 'replacement', mask_t.numpy().astype(bool)

trig_type, trig_data = _make_trigger()
trig_t = torch.from_numpy(trig_data).to(device) if trig_type == 'additive' else None
print(f'Trigger type={trig_type}', flush=True)

def apply_trigger_images(images_gpu):
    """Apply trigger to a batch of images. Returns triggered images."""
    if trig_type == 'additive':
        return torch.clamp(images_gpu + trig_t.unsqueeze(0), -1.0, 1.0)
    else:  # replacement
        mask_b = torch.from_numpy(trig_data).to(images_gpu.device).bool()
        mask_b = mask_b.unsqueeze(0).expand_as(images_gpu)
        return torch.where(mask_b, torch.tensor(1.0, device=images_gpu.device), images_gpu)

def apply_trigger_batch(images, poison_mask):
    """Patch train_mod.apply_pattern_batch_gpu for client training."""
    pm_exp = poison_mask.view(-1, 1, 1, 1)
    if trig_type == 'additive':
        pat = trig_t.to(images.device)
        poisoned = torch.clamp(images + pat.unsqueeze(0), -1.0, 1.0)
    else:
        mask_b = torch.from_numpy(trig_data).to(images.device).bool()
        mask_b = mask_b.unsqueeze(0).expand_as(images)
        poisoned = torch.where(mask_b, torch.tensor(1.0, device=images.device), images)
    labels = torch.full((images.size(0),), TARGET, device=images.device, dtype=torch.long)
    return (torch.where(pm_exp.expand_as(images).bool(), poisoned, images), labels)

# ── Flatten helpers ───────────────────────────────────────────────────────────
_BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')

def is_param(key):
    return not any(key.endswith(b) for b in _BUFFER_SUFFIXES)

def flatten_params_only(state_dict):
    """Flatten trainable parameters only (no BN buffers) → numpy float32."""
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for k, v in state_dict.items() if is_param(k)])

def flatten_all(state_dict):
    """Flatten all keys including BN buffers → numpy float32."""
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for v in state_dict.values()])

def reconstruct_state(ref_state, params_delta, kept_states_for_buffers=None):
    """Apply params_delta (numpy, params-only) back into a full state dict.
    BN buffers are averaged from kept_states_for_buffers (or copied from ref)."""
    new_state = {}
    offset = 0
    for k, v in ref_state.items():
        if is_param(k):
            numel = v.numel()
            chunk = params_delta[offset:offset + numel].reshape(v.shape)
            new_state[k] = v.cpu().float() + torch.from_numpy(chunk.astype(np.float32))
            offset += numel
        else:
            if kept_states_for_buffers:
                avg = torch.stack([w[k].float().cpu()
                                   for w in kept_states_for_buffers]).mean(0)
                new_state[k] = avg
            else:
                new_state[k] = v.cpu()
    return new_state

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(state):
    """ASR, true clean_acc (full test set), non_target_tnr."""
    m = ResNet8(num_classes=N_CLASSES).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    with torch.no_grad():
        # ASR: triggered non-target images → target class
        trig_imgs = apply_trigger_images(non_target_gpu)
        asr = torch.softmax(m(trig_imgs), 1).mean(0)[TARGET].item()

        # True clean_acc: correct-class accuracy on full test set
        preds_all = torch.softmax(m(all_imgs_gpu), 1).argmax(1)
        clean_acc = (preds_all == all_lbls_gpu).float().mean().item()

        # non_target_tnr: fraction of non-target images not classified as target
        preds_nt = preds_all[all_lbls_gpu != TARGET]
        non_target_tnr = (preds_nt != TARGET).float().mean().item()

    return asr, clean_acc, non_target_tnr

# ── Client training ───────────────────────────────────────────────────────────
ADV_SET = set(handle.adversarial_namelist)

def train_client(state, client_id, round_seed):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    is_adv = client_id in ADV_SET
    p = copy.deepcopy(params)
    p['poison_rate'] = args.poison_rate

    import train as tm
    _orig = tm.apply_pattern_batch_gpu
    if is_adv:
        def patched(images, pms, index, handle_, poison_mask,
                    _fn=apply_trigger_batch):
            return _fn(images, poison_mask)
        tm.apply_pattern_batch_gpu = patched

    mod = ResNet8(num_classes=N_CLASSES).to(device)
    mod.load_state_dict({k: v.to(device) for k, v in state.items()})
    try:
        w_local = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id, params=p, model=mod,
            dataset=handle.train_data[client_id], handle=handle,
            poison=is_adv,
        )
    finally:
        if is_adv:
            tm.apply_pattern_batch_gpu = _orig
    return {k: v.cpu() for k, v in w_local.items()}

# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg_agg(state, local_states, client_ids, sizes):
    total = sum(sizes)
    g = flatten_params_only(state)
    deltas = np.stack([flatten_params_only(w) - g for w in local_states])
    weights = np.array(sizes, dtype=np.float64) / total
    agg = (deltas * weights[:, None]).sum(axis=0)
    return reconstruct_state(state, agg.astype(np.float32), local_states)

# ── FLTrust ───────────────────────────────────────────────────────────────────
# Build root dataset once (10 images per class from test set)
_root_imgs, _root_lbls = [], []
for cls in range(N_CLASSES):
    idx = (all_lbls == cls).nonzero(as_tuple=True)[0][:args.root_per_class]
    _root_imgs.append(all_imgs[idx])
    _root_lbls.append(all_lbls[idx])
root_imgs_gpu = torch.cat(_root_imgs).to(device)
root_lbls_gpu = torch.cat(_root_lbls).to(device)
print(f'Root dataset: {root_imgs_gpu.shape}', flush=True)

def fltrust_server_update(global_state):
    """Train server model on root dataset for local_epochs steps → Δw_s (params only)."""
    srv = ResNet8(num_classes=N_CLASSES).to(device)
    srv.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    srv.train()
    opt = optim.SGD(srv.parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=params.get('weight_decay', 1e-4))
    ds = TensorDataset(root_imgs_gpu, root_lbls_gpu)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for _ in range(args.local_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(srv(xb), yb).backward()
            opt.step()
    w_s = {k: v.cpu() for k, v in srv.state_dict().items()}
    g_s = flatten_params_only(w_s) - flatten_params_only(global_state)
    return torch.from_numpy(g_s)

def fltrust_agg(state, local_states, client_ids, sizes):
    """Returns (new_state, per_client_info_list).
    per_client_info_list: list of dicts with trust_score, norm_params, norm_buffers.
    Step 2b: logs ||Δ_params||₂ and ||Δ_buffers||₂ per client."""
    g_flat = flatten_params_only(state)

    # Server update (params only, correct epoch count)
    g_s_t = fltrust_server_update(state)
    g_s_norm = g_s_t.norm().item()

    client_info = []
    deltas_params = []
    norm_sums = []

    for i, (cid, w_loc) in enumerate(zip(client_ids, local_states)):
        # Params-only delta
        g_i_params = flatten_params_only(w_loc) - g_flat
        g_i_t = torch.from_numpy(g_i_params)

        # Buffer-only delta (for Step 2b logging)
        def _buf_norm(sd_a, sd_b):
            parts = []
            for k in sd_a:
                if not is_param(k):
                    parts.append((sd_a[k].float().cpu() - sd_b[k].float().cpu()).flatten())
            return torch.cat(parts).norm().item() if parts else 0.0

        norm_params = g_i_t.norm().item()
        norm_buffers = _buf_norm(w_loc, state)

        cos_raw = F.cosine_similarity(g_i_t.unsqueeze(0),
                                      g_s_t.unsqueeze(0)).item()
        ts = max(0.0, cos_raw)

        deltas_params.append(g_i_params)
        norm_sums.append(ts)
        client_info.append({
            'client': cid,
            'role': 'ADV' if cid in ADV_SET else 'BEN',
            'defense_score': ts,
            'norm_params': norm_params,
            'norm_buffers': norm_buffers,
        })

    # Weighted aggregation (trust-score weighted, normalised)
    ts_arr = np.array(norm_sums, dtype=np.float64)
    total_ts = ts_arr.sum() + 1e-9
    deltas_stack = np.stack(deltas_params)
    agg = (deltas_stack * ts_arr[:, None] / total_ts).sum(axis=0)

    # BN buffers: plain average from all local states (trust score not applied)
    new_state = reconstruct_state(state, agg.astype(np.float32), local_states)
    return new_state, client_info

# ── FoolsGold ─────────────────────────────────────────────────────────────────
class FoolsGold:
    def __init__(self, all_ids):
        self.ids = list(all_ids)
        self.n = len(self.ids)
        self.contrib = None  # (n, D) — params only

    def aggregate(self, state, local_states, client_ids, sizes):
        g_flat = flatten_params_only(state)
        deltas = np.stack([flatten_params_only(w) - g_flat for w in local_states])
        D = deltas.shape[1]

        if self.contrib is None:
            self.contrib = np.zeros((self.n, D), dtype=np.float32)

        cidxs = [self.ids.index(cid) for cid in client_ids]
        for k, cidx in enumerate(cidxs):
            norm = np.linalg.norm(deltas[k]) + 1e-9
            self.contrib[cidx] += deltas[k] / norm

        bc = self.contrib[cidxs]
        norms = np.linalg.norm(bc, axis=1, keepdims=True) + 1e-9
        normed = bc / norms
        cs = normed @ normed.T

        n_b = len(client_ids)
        v = np.zeros(n_b)
        for i in range(n_b):
            sims = [cs[i, j] for j in range(n_b) if j != i]
            v[i] = max(sims) if sims else 0.0

        lr = np.clip(1.0 - v, 0.0, 1.0)
        if lr.max() > 1e-9:
            lr = lr / lr.max()
        else:
            lr = np.ones(n_b) / n_b

        total_lr = lr.sum() + 1e-9
        agg = (deltas * lr[:, None]).sum(axis=0) / total_lr

        new_state = reconstruct_state(state, agg.astype(np.float32), local_states)
        client_info = []
        for i, cid in enumerate(client_ids):
            client_info.append({
                'client': cid,
                'role': 'ADV' if cid in ADV_SET else 'BEN',
                'defense_score': float(lr[i]),
            })
        return new_state, client_info

# ── FLAME ─────────────────────────────────────────────────────────────────────
def flame_agg(state, local_states, client_ids, sizes, lam, rng):
    n = len(local_states)
    g_flat = flatten_params_only(state)
    deltas = np.stack([flatten_params_only(w) - g_flat for w in local_states])

    norms = np.linalg.norm(deltas, axis=1, keepdims=True) + 1e-9
    normed = (deltas / norms)
    cos_dist = np.clip(1.0 - normed @ normed.T, 0.0, 2.0).astype(np.float64)

    min_cs = max(2, n // 2 + 1)
    try:
        labels = HDBSCAN(min_cluster_size=min_cs, metric='precomputed',
                         allow_single_cluster=True).fit_predict(cos_dist)
    except Exception:
        labels = np.zeros(n, dtype=int)

    if labels.max() >= 0:
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest = unique[counts.argmax()]
        keep = labels == largest
    else:
        keep = np.ones(n, dtype=bool)

    kept_deltas = deltas[keep]
    kept_norms  = np.linalg.norm(kept_deltas, axis=1)
    S_t = float(np.median(kept_norms))
    clipped = np.stack([
        d / max(1.0, float(np.linalg.norm(d)) / (S_t + 1e-9))
        for d in kept_deltas
    ])

    agg = clipped.mean(axis=0)
    sigma = lam * S_t
    if sigma > 0:
        agg = agg + rng.normal(0.0, sigma, agg.shape).astype(np.float32)

    kept_local = [local_states[i] for i in range(n) if keep[i]]
    new_state = reconstruct_state(state, agg.astype(np.float32), kept_local)

    client_info = []
    for i, cid in enumerate(client_ids):
        excluded = not keep[i]
        client_info.append({
            'client': cid,
            'role': 'ADV' if cid in ADV_SET else 'BEN',
            'defense_score': 0.0 if excluded else 1.0,
            'excluded': int(excluded),
            'S_t': S_t,
            'sigma': sigma,
        })
    return new_state, client_info

# ── Main loop ─────────────────────────────────────────────────────────────────
fg_obj = FoolsGold(handle.participants_list) if args.defense == 'foolsgold' else None
rng_flame = np.random.RandomState(args.seed)

COLS = ['defense', 'trigger', 'alpha', 'round', 'client', 'role',
        'defense_score', 'asr', 'clean_acc', 'non_target_tnr',
        'norm_params', 'norm_buffers', 'excluded', 'S_t', 'sigma']
all_rows = []

# Checkpoint file for resuming
tag = f'{args.defense}_{args.trigger}_a{args.alpha}'
ckpt_out = os.path.join(args.out, f'step5_{tag}_ckpt.pt')
start_round = 1
if os.path.exists(ckpt_out):
    saved = torch.load(ckpt_out, map_location='cpu')
    global_state = saved['global_state']
    start_round  = saved['round'] + 1
    all_rows     = saved.get('rows', [])
    if fg_obj is not None and 'fg_contrib' in saved:
        fg_obj.contrib = saved['fg_contrib']
    print(f'Resumed from round {start_round - 1}', flush=True)

print(f'\n{"rnd":>4}  {"excl/TS":>10}  {"ASR":>7}  {"ACC":>7}  {"TNR":>7}')
print('-' * 45)

for rnd in range(start_round, args.n_rounds + 1):
    # Client selection (reproducible: seed per round)
    rng_sel = random.Random(args.seed * 100000 + rnd)
    selected = rng_sel.sample(handle.participants_list, args.n_clients)

    local_states, sizes, client_ids = [], [], []
    for cid in selected:
        w = train_client(global_state, cid, round_seed=args.seed * 10000 + rnd * 100 + cid)
        local_states.append(w)
        sizes.append(handle.clients_data_num[cid])
        client_ids.append(cid)

    # Aggregation
    if args.defense == 'fedavg':
        global_state = fedavg_agg(global_state, local_states, client_ids, sizes)
        client_info = [{'client': cid,
                        'role': 'ADV' if cid in ADV_SET else 'BEN',
                        'defense_score': sizes[i] / sum(sizes)}
                       for i, cid in enumerate(client_ids)]
    elif args.defense == 'fltrust':
        global_state, client_info = fltrust_agg(global_state, local_states,
                                                 client_ids, sizes)
    elif args.defense == 'foolsgold':
        global_state, client_info = fg_obj.aggregate(global_state, local_states,
                                                      client_ids, sizes)
    elif args.defense == 'flame':
        global_state, client_info = flame_agg(global_state, local_states,
                                              client_ids, sizes,
                                              args.lambda_flame, rng_flame)

    asr, clean_acc, non_target_tnr = compute_metrics(global_state)

    # Summarise ADV exclusion/score for console
    adv_in_round = [ci for ci in client_info if ci['role'] == 'ADV']
    ben_in_round = [ci for ci in client_info if ci['role'] == 'BEN']
    adv_score_m = np.mean([ci['defense_score'] for ci in adv_in_round]) if adv_in_round else float('nan')
    ben_score_m = np.mean([ci['defense_score'] for ci in ben_in_round]) if ben_in_round else float('nan')
    print(f'{rnd:>4}  ADV={adv_score_m:>5.3f}/BEN={ben_score_m:>5.3f}  '
          f'{asr:>7.4f}  {clean_acc:>7.4f}  {non_target_tnr:>7.4f}', flush=True)

    for ci in client_info:
        row = {
            'defense':       args.defense,
            'trigger':       args.trigger,
            'alpha':         args.alpha,
            'round':         rnd,
            'client':        ci['client'],
            'role':          ci['role'],
            'defense_score': ci.get('defense_score', float('nan')),
            'asr':           asr,
            'clean_acc':     clean_acc,
            'non_target_tnr': non_target_tnr,
            'norm_params':   ci.get('norm_params', ''),
            'norm_buffers':  ci.get('norm_buffers', ''),
            'excluded':      ci.get('excluded', ''),
            'S_t':           ci.get('S_t', ''),
            'sigma':         ci.get('sigma', ''),
        }
        all_rows.append(row)

    if rnd % args.checkpoint_every == 0:
        ckpt_data = {'round': rnd, 'global_state': global_state, 'rows': all_rows}
        if fg_obj is not None:
            ckpt_data['fg_contrib'] = fg_obj.contrib
        torch.save(ckpt_data, ckpt_out)
        print(f'  [checkpoint saved at round {rnd}]', flush=True)

# ── Final CSV ─────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out, f'step5_{tag}.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(all_rows)
print(f'\nSaved to {csv_path}', flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
last_100 = [r for r in all_rows if r['round'] > args.n_rounds - 100
            and r['client'] == client_ids[0]]
asr_vals = [r['asr'] for r in last_100]
acc_vals = [r['clean_acc'] for r in last_100]
print(f'\n=== Final 100 rounds: ASR={np.mean(asr_vals):.4f}±{np.std(asr_vals):.4f}  '
      f'clean_acc={np.mean(acc_vals):.4f}±{np.std(acc_vals):.4f} ===')

adv_rows = [r for r in all_rows if r['role'] == 'ADV' and r['round'] > args.n_rounds - 100]
ben_rows = [r for r in all_rows if r['role'] == 'BEN' and r['round'] > args.n_rounds - 100]
if adv_rows:
    adv_scores = [r['defense_score'] for r in adv_rows if r['defense_score'] != '']
    ben_scores = [r['defense_score'] for r in ben_rows if r['defense_score'] != '']
    print(f'  Last 100 rounds: ADV score={np.mean(adv_scores):.4f}  '
          f'BEN score={np.mean(ben_scores):.4f}')
