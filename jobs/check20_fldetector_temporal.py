#!/usr/bin/env python3
"""Check 20: FLDetector temporal distance measurement.

FLDetector (USENIX Security 2022) detects malicious clients by tracking
temporal consistency of per-client weight updates:

  dist_i^t   = ||Δw_i^t - Δw_i^{t-1}||₂
  score_i    = mean(dist_i^{t-W+1 .. t})   (W=10 round rolling mean)

High score → erratic updates → flagged as malicious.
The L-BFGS HVP version refines Δw_i^{t-1} to a predicted value, but the
paper shows this adds marginal AUC over the simpler ||Δw^t - Δw^{t-1}|| form.
We use the simpler form to avoid porting MXNet.

Prediction: our cov_64_pos trigger (which evades FilterFL) produces
consistent updates round-over-round because:
  (a) the trigger is fixed — same perturbation every round
  (b) the model updates from the same local distribution each round
→ temporal distance should be similar to benign clients.

FLDetector was designed to catch scaling attacks (×100 amplification) that
produce huge, erratic jumps — not fixed-pattern backdoors.

Setup:
  N_ROUNDS = 35   (FLDetector needs ~20 warmup + 10-round window)
  Clients selected EVERY round (needed for clean temporal tracking):
    Adversarial: [34, 36, 47, 60] — trigger=cov_64_pos
    Benign:      [0, 1, 2, 3, 5, 10]
  Aggregation: FedAvg weighted by dataset size
  Poison rates: 0.50 and 0.10 (separate runs)

Output: per-round temporal distances + rolling malicious_score for each
client, plus a summary comparing adversarial vs benign score distributions.
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
ap.add_argument('--window',       type=int, default=10)
ap.add_argument('--poison-rates', type=float, nargs='+', default=[0.50, 0.10])
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

# ── Trigger (cov_64_pos) ─────────────────────────────────────────────────────
rng64 = np.random.RandomState(42)
pos64 = rng64.choice(H * W, size=64, replace=False)
trigger_np = np.zeros((C, H, W), dtype=np.float32)
for p in pos64:
    h_p, w_p = divmod(int(p), W)
    trigger_np[:, h_p, w_p] = args.amplitude
trigger_t = torch.from_numpy(trigger_np)
print(f'Trigger: cov_64_pos  L∞={np.abs(trigger_np).max():.2f}', flush=True)

# ── Flatten helper ────────────────────────────────────────────────────────────
def flatten(state_dict):
    return torch.cat([v.float().cpu().flatten() for v in state_dict.values()])

# ── FedAvg aggregation ────────────────────────────────────────────────────────
def fedavg(global_state, local_states, sizes):
    total = sum(sizes)
    new_state = {}
    for k in global_state:
        weighted = sum(local_states[i][k].float().cpu() * sizes[i] / total
                       for i in range(len(local_states)))
        new_state[k] = weighted
    return new_state

# ── Client training ───────────────────────────────────────────────────────────
def train_client(global_state, client_id, poison, poison_rate, round_seed):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    p = copy.deepcopy(params)
    p['poison_rate'] = poison_rate

    import train as tm
    _orig = tm.apply_pattern_batch_gpu

    if poison:
        _pt = trigger_t
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
            client=client_id,
            params=p,
            model=m,
            dataset=handle.train_data[client_id],
            handle=handle,
            poison=poison,
        )
    finally:
        if poison:
            tm.apply_pattern_batch_gpu = _orig

    return {k: v.cpu() for k, v in w_local.items()}

# ── ASR probe ─────────────────────────────────────────────────────────────────
def asr_proxy(w_local):
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict({k: v.to(device) for k, v in w_local.items()})
    m.eval()
    with torch.no_grad():
        trig_imgs = torch.clamp(non_target + trigger_t.to(device).unsqueeze(0),
                                -1.0, 1.0)
        return torch.softmax(m(trig_imgs), 1).mean(0)[TARGET].item()

# ── Main loop per poison rate ──────────────────────────────────────────────────
ALL_CLIENTS = args.adv_clients + args.ben_clients
ADV_SET = set(args.adv_clients)

ROUND_COLS = ['poison_rate', 'round', 'client', 'role',
              'delta_norm', 'temporal_dist', 'rolling_score', 'asr']
round_rows = []

for poison_rate in args.poison_rates:
    print(f'\n{"="*70}')
    print(f'Poison rate: {poison_rate:.2f}')
    print(f'{"="*70}')

    # Reset to warmup checkpoint
    handle.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
    g_flat = flatten(global_state)

    # Per-client history of Δw (flattened)
    prev_delta = {cid: None for cid in ALL_CLIENTS}
    dist_history = {cid: [] for cid in ALL_CLIENTS}

    print(f'\n{"round":>5}  {"client":>6}  {"role":>6}  '
          f'{"Δw_norm":>8}  {"temp_dist":>10}  {"roll_score":>11}  {"asr":>7}')
    print('-' * 62)

    for rnd in range(1, args.n_rounds + 1):
        local_states = []
        local_sizes  = []

        for cid in ALL_CLIENTS:
            is_adv = cid in ADV_SET
            round_seed = rnd * 1000 + cid
            w_local = train_client(
                global_state, cid,
                poison=is_adv, poison_rate=poison_rate,
                round_seed=round_seed
            )
            local_states.append(w_local)
            local_sizes.append(handle.clients_data_num[cid])

            # Compute Δw^t = w_local - w_global
            delta = flatten(w_local) - flatten(global_state)
            delta_norm = delta.norm().item()

            # Temporal distance
            if prev_delta[cid] is not None:
                temp_dist = (delta - prev_delta[cid]).norm().item()
            else:
                temp_dist = float('nan')

            prev_delta[cid] = delta.clone()
            if not np.isnan(temp_dist):
                dist_history[cid].append(temp_dist)

            # Rolling malicious_score (mean of last W rounds)
            if len(dist_history[cid]) >= args.window:
                rolling = float(np.mean(dist_history[cid][-args.window:]))
            elif dist_history[cid]:
                rolling = float(np.mean(dist_history[cid]))
            else:
                rolling = float('nan')

            # ASR only every 5 rounds (cheap enough)
            asr_val = asr_proxy(w_local) if rnd % 5 == 0 or rnd == 1 else float('nan')

            role = 'ADV' if is_adv else 'BEN'
            print(f'{rnd:>5}  {cid:>6}  {role:>6}  '
                  f'{delta_norm:>8.3f}  '
                  f'{temp_dist if not np.isnan(temp_dist) else float("nan"):>10.4f}  '
                  f'{rolling if not np.isnan(rolling) else float("nan"):>11.4f}  '
                  f'{asr_val if not np.isnan(asr_val) else float("nan"):>7.4f}',
                  flush=True)

            round_rows.append({
                'poison_rate': poison_rate, 'round': rnd, 'client': cid,
                'role': role, 'delta_norm': delta_norm,
                'temporal_dist': temp_dist if not np.isnan(temp_dist) else '',
                'rolling_score': rolling if not np.isnan(rolling) else '',
                'asr': asr_val if not np.isnan(asr_val) else '',
            })

        # FedAvg update
        global_state = fedavg(global_state, local_states, local_sizes)

    # ── Summary for this poison rate ─────────────────────────────────────────
    print(f'\n=== Final malicious_score (mean of last {args.window} rounds) ===')
    print(f'{"client":>8}  {"role":>6}  {"score_mean":>11}  {"score_std":>10}')
    print('-' * 40)
    for cid in ALL_CLIENTS:
        hist = dist_history[cid]
        last_w = hist[-args.window:] if len(hist) >= args.window else hist
        role = 'ADV' if cid in ADV_SET else 'BEN'
        print(f'{cid:>8}  {role:>6}  {np.mean(last_w):>11.4f}  {np.std(last_w):>10.4f}')

    adv_scores = [np.mean(dist_history[c][-args.window:])
                  for c in args.adv_clients if len(dist_history[c]) >= args.window]
    ben_scores = [np.mean(dist_history[c][-args.window:])
                  for c in args.ben_clients if len(dist_history[c]) >= args.window]
    if adv_scores and ben_scores:
        print(f'\n  ADV mean score: {np.mean(adv_scores):.4f} ± {np.std(adv_scores):.4f}')
        print(f'  BEN mean score: {np.mean(ben_scores):.4f} ± {np.std(ben_scores):.4f}')
        ratio = np.mean(adv_scores) / (np.mean(ben_scores) + 1e-9)
        print(f'  ADV/BEN ratio: {ratio:.3f}  '
              f'(>2 → discriminative; ≈1 → FLDetector blind)')

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out, 'check20_fldetector_temporal.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=ROUND_COLS)
    wr.writeheader(); wr.writerows(round_rows)
print(f'\nSaved to {csv_path}')
