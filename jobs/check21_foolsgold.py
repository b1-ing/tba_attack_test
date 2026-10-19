#!/usr/bin/env python3
"""Check 21: Foolsgold defense against same-trigger vs diverse-trigger attack.

Foolsgold (AAAI 2020) penalises clients with correlated update histories:
  1. Track cumulative normalised gradient per client: contrib_i += g_i/||g_i||
  2. Pairwise cosine similarity: cs[i,j] = cos(contrib_i, contrib_j)
  3. Sycophancy score: v[i] = max_{j≠i} cs[i,j]
  4. Weight: lr[i] = (1 - v[i]) / max_j(1 - v[j])   (re-normalised to [0,1])
  5. Aggregate: Δw_global = Σ_i lr[i] * Δw_i / Σ_i lr[i]

Vulnerability for our attack: all 4 adversarial clients use the SAME trigger
→ highly correlated updates → all get lr ≈ 0 → backdoor neutralised.

Fix (diverse-trigger variant): each adversarial client uses a DIFFERENT
random 64px mask (but same amplitude). Updates become decorrelated while
each trigger independently evades FilterFL's coverage threshold.

This check runs 3 conditions × 35 rounds:
  (A) FedAvg      — standard aggregation (baseline, from check20)
  (B) FoolsGold, same trigger  — all ADV use cov_64_pos (same mask)
  (C) FoolsGold, diverse trigger — each ADV client gets its own random 64px mask

Metrics per round: foolsgold_weight per client, ASR under global model,
feat_align on the aggregated global model.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',     default='FilterFL')
ap.add_argument('--warmup-dir',   default=None)
ap.add_argument('--out',          default='.')
ap.add_argument('--adv-clients',  type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--ben-clients',  type=int, nargs='+', default=[0, 1, 2, 3, 5, 10])
ap.add_argument('--n-rounds',     type=int, default=35)
ap.add_argument('--poison-rate',  type=float, default=0.50)
ap.add_argument('--amplitude',    type=float, default=0.5)
ap.add_argument('--n-test',       type=int, default=500)
ap.add_argument('--alpha',        type=float, default=0.5)
ap.add_argument('--global-seed',  type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.nn.functional as F
import yaml

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

# ── Trigger patterns ──────────────────────────────────────────────────────────
def make_sparse(positions):
    pat = np.zeros((C, H, W), dtype=np.float32)
    for p in positions:
        h_p, w_p = divmod(int(p), W)
        pat[:, h_p, w_p] = args.amplitude
    return pat

# Shared trigger (cov_64_pos — same mask for all clients)
rng_shared = np.random.RandomState(42)
shared_pos = rng_shared.choice(H * W, size=64, replace=False)
shared_trigger = make_sparse(shared_pos)

# Per-client diverse triggers (different mask per adversarial client)
diverse_triggers = {}
for i, cid in enumerate(args.adv_clients):
    rng_div = np.random.RandomState(100 + i)
    pos = rng_div.choice(H * W, size=64, replace=False)
    diverse_triggers[cid] = make_sparse(pos)
    print(f'  Diverse trigger for client {cid}: '
          f'overlap with shared={len(set(pos) & set(shared_pos))} px')

# ── Flatten helper ─────────────────────────────────────────────────────────────
def flatten(state_dict):
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for v in state_dict.values()])

# ── Foolsgold aggregation ──────────────────────────────────────────────────────
class FoolsGold:
    def __init__(self, all_client_ids):
        self.ids = list(all_client_ids)
        self.n = len(self.ids)
        self.contrib = None   # (n, d) — initialised on first call

    def aggregate(self, global_state, local_states, client_ids, sizes):
        """Return new global_state and per-client Foolsgold weights."""
        deltas = []
        for cid in client_ids:
            idx_in_batch = client_ids.index(cid)
            d = flatten(local_states[idx_in_batch]) - flatten(global_state)
            deltas.append(d)
        deltas = np.stack(deltas)          # (batch, D)
        D = deltas.shape[1]

        if self.contrib is None:
            self.contrib = np.zeros((self.n, D), dtype=np.float32)

        # Update cumulative normalised contributions
        client_indices = [self.ids.index(cid) for cid in client_ids]
        for k, cidx in enumerate(client_indices):
            norm = np.linalg.norm(deltas[k]) + 1e-9
            self.contrib[cidx] += deltas[k] / norm

        # Pairwise cosine similarity over selected clients only
        batch_contrib = self.contrib[client_indices]    # (batch, D)
        norms = np.linalg.norm(batch_contrib, axis=1, keepdims=True) + 1e-9
        normed = batch_contrib / norms                  # (batch, D)
        cs = normed @ normed.T                          # (batch, batch)

        # Sycophancy score: max similarity to any OTHER selected client
        n_batch = len(client_ids)
        v = np.zeros(n_batch)
        for i in range(n_batch):
            sims = [cs[i, j] for j in range(n_batch) if j != i]
            v[i] = max(sims) if sims else 0.0

        # Foolsgold weights: (1 - v), re-normalised
        lr = np.clip(1.0 - v, 0.0, 1.0)
        if lr.max() > 1e-9:
            lr = lr / lr.max()
        else:
            lr = np.ones(n_batch) / n_batch

        # Weighted aggregation of Δw
        total = lr.sum() + 1e-9
        agg_delta = (deltas * lr[:, None]).sum(axis=0) / total   # (D,)

        # Reconstruct new global state dict
        new_state = {}
        offset = 0
        for k, v_t in global_state.items():
            numel = v_t.numel()
            chunk = agg_delta[offset:offset + numel].reshape(v_t.shape)
            new_state[k] = v_t.cpu().float() + torch.from_numpy(chunk)
            offset += numel

        weights = {cid: float(lr[k]) for k, cid in enumerate(client_ids)}
        return new_state, weights

# ── FedAvg aggregation ─────────────────────────────────────────────────────────
def fedavg(global_state, local_states, client_ids, sizes):
    total = sum(sizes)
    new_state = {}
    for k in global_state:
        weighted = sum(local_states[i][k].float().cpu() * sizes[i] / total
                       for i in range(len(local_states)))
        new_state[k] = weighted
    weights = {cid: sizes[i] / total for i, cid in enumerate(client_ids)}
    return new_state, weights

# ── Client training ────────────────────────────────────────────────────────────
def train_client(global_state, client_id, trigger_np, round_seed):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    p = copy.deepcopy(params)
    p['poison_rate'] = args.poison_rate
    import train as tm
    _orig = tm.apply_pattern_batch_gpu
    is_adv = trigger_np is not None
    if is_adv:
        _pt = torch.from_numpy(trigger_np)
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

# ── ASR on global model ───────────────────────────────────────────────────────
def global_metrics(global_state, trigger_np):
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in global_state.items()})
    m.eval()
    trig_t = torch.from_numpy(trigger_np).to(device)
    with torch.no_grad():
        trig_imgs = torch.clamp(non_target + trig_t.unsqueeze(0), -1.0, 1.0)
        asr = torch.softmax(m(trig_imgs), 1).mean(0)[TARGET].item()
        # Clean accuracy on non-target images (sanity check)
        clean_acc = (torch.softmax(m(non_target), 1).argmax(1) != TARGET).float().mean().item()
    return asr, clean_acc

# ── Main: three conditions ────────────────────────────────────────────────────
ALL_CLIENTS = args.adv_clients + args.ben_clients
ADV_SET = set(args.adv_clients)

CONDITIONS = [
    ('fedavg_same',   'FedAvg + same trigger',    False, 'same'),
    ('fg_same',       'FoolsGold + same trigger',  True,  'same'),
    ('fg_diverse',    'FoolsGold + diverse trigger',True, 'diverse'),
]

COLS = ['condition', 'round', 'client', 'role', 'fg_weight', 'asr_global', 'clean_acc']
all_rows = []

for cond_key, cond_label, use_fg, trig_mode in CONDITIONS:
    print(f'\n{"="*70}')
    print(f'Condition: {cond_label}')
    print(f'{"="*70}')

    handle.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}

    fg = FoolsGold(ALL_CLIENTS) if use_fg else None

    print(f'\n{"round":>5}  {"client":>6}  {"role":>6}  '
          f'{"fg_w":>7}  {"asr":>7}  {"clean":>7}')
    print('-' * 47)

    for rnd in range(1, args.n_rounds + 1):
        local_states, sizes, client_ids_rnd = [], [], []

        for cid in ALL_CLIENTS:
            is_adv = cid in ADV_SET
            if is_adv:
                if trig_mode == 'same':
                    trig = shared_trigger
                else:
                    trig = diverse_triggers[cid]
            else:
                trig = None

            w_local = train_client(global_state, cid, trig,
                                   round_seed=rnd * 1000 + cid)
            local_states.append(w_local)
            sizes.append(handle.clients_data_num[cid])
            client_ids_rnd.append(cid)

        if use_fg:
            global_state, fg_weights = fg.aggregate(
                global_state, local_states, client_ids_rnd, sizes)
        else:
            global_state, fg_weights = fedavg(
                global_state, local_states, client_ids_rnd, sizes)

        # Global metrics every 5 rounds
        if rnd % 5 == 0 or rnd == 1:
            ref_trig = shared_trigger if trig_mode == 'same' else diverse_triggers[args.adv_clients[0]]
            torch.manual_seed(42); np.random.seed(42)
            asr_g, fa_g = global_metrics(global_state, ref_trig)
        else:
            asr_g, fa_g = float('nan'), float('nan')

        for cid in ALL_CLIENTS:
            role = 'ADV' if cid in ADV_SET else 'BEN'
            w = fg_weights.get(cid, float('nan'))
            print(f'{rnd:>5}  {cid:>6}  {role:>6}  '
                  f'{w:>7.4f}  '
                  f'{asr_g if not np.isnan(asr_g) else float("nan"):>7.4f}  '
                  f'{fa_g if not np.isnan(fa_g) else float("nan"):>7.4f}',
                  flush=True)
            all_rows.append({
                'condition': cond_key, 'round': rnd, 'client': cid, 'role': role,
                'fg_weight': w,
                'asr_global': asr_g if not np.isnan(asr_g) else '',
                'clean_acc': fa_g if not np.isnan(fa_g) else '',
            })

    # Final summary
    print(f'\n=== Final round Foolsgold weights ===')
    adv_ws = [fg_weights[c] for c in args.adv_clients]
    ben_ws = [fg_weights[c] for c in args.ben_clients]
    print(f'  ADV weights: {[f"{w:.4f}" for w in adv_ws]}  mean={np.mean(adv_ws):.4f}')
    print(f'  BEN weights: {[f"{w:.4f}" for w in ben_ws]}  mean={np.mean(ben_ws):.4f}')
    ref_trig = shared_trigger if trig_mode == 'same' else diverse_triggers[args.adv_clients[0]]
    torch.manual_seed(42); np.random.seed(42)
    asr_f, fa_f = global_metrics(global_state, ref_trig)
    print(f'  Final ASR={asr_f:.4f}  clean_acc={fa_f:.4f}')

# ── Summary table ──────────────────────────────────────────────────────────────
print('\n\n=== Cross-condition summary (round 35 metrics) ===')
print(f'{"condition":>18}  {"ADV_w":>7}  {"BEN_w":>7}  {"ASR":>7}  {"fa_g":>7}')
print('-' * 55)
for cond_key, cond_label, _, _ in CONDITIONS:
    rnd35 = [r for r in all_rows if r['condition'] == cond_key and r['round'] == 35]
    adv_w = np.mean([r['fg_weight'] for r in rnd35 if r['role'] == 'ADV'])
    ben_w = np.mean([r['fg_weight'] for r in rnd35 if r['role'] == 'BEN'])
    # Find last asr/fa for this condition
    asr_rows = [r for r in all_rows if r['condition'] == cond_key
                and r['asr_global'] != '' and r['round'] <= 35]
    asr_val = float(asr_rows[-1]['asr_global']) if asr_rows else float('nan')
    ca_val  = float(asr_rows[-1]['clean_acc'])  if asr_rows else float('nan')
    print(f'{cond_label:>18}  {adv_w:>7.4f}  {ben_w:>7.4f}  '
          f'{asr_val:>7.4f}  {ca_val:>7.4f}')

csv_path = os.path.join(args.out, 'check21_foolsgold.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(all_rows)
print(f'\nSaved to {csv_path}')
