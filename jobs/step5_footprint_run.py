#!/usr/bin/env python3
"""Steps 2a + 2b: weight-footprint paired training and backdoor-direction analysis.

60-round FedAvg from warmup checkpoint. At every round where an adversarial
client is selected, trains that client twice (poisoned and clean, same seed)
to measure the trigger-only update component.

Outputs (in --out):
  step5_footprint_{trigger}.csv         per-round: asr, clean_acc, cos projections
  step5_footprint_{trigger}_paired.csv  per-(round,client): footprint ratios
  step5_footprint_{trigger}_d.npy       d_hat direction vector (params-only, flattened)
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--trigger',     required=True, choices=['checker', 'patch'])
ap.add_argument('--code-dir',    default='FilterFL')
ap.add_argument('--warmup-dir',  default=None)
ap.add_argument('--out',         default='.')
ap.add_argument('--n-rounds',    type=int,   default=60)
ap.add_argument('--seed',        type=int,   default=1)
ap.add_argument('--alpha',       type=float, default=0.5)
ap.add_argument('--amplitude',   type=float, default=0.5)
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

params['seed']                         = args.seed
params['dirichlet_alpha']              = args.alpha
params['defence_method']               = 'None'
params['number_of_total_participants'] = 100
params['num_clients']                  = 10
params['poison_client_num']            = 10
params['local_epochs']                 = 2
params['lr']                           = 0.01
params['momentum']                     = 0.9
params['poison_rate']                  = 0.5
params['is_poison']                    = True
ref_main.seed(args.seed)

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET    = handle.target
N_CLASSES = 10
C, H, W   = 3, 32, 32
ADV_SET   = set(handle.adversarial_namelist)
print(f'Target: {TARGET}  Trigger: {args.trigger}  ADV: {sorted(ADV_SET)}', flush=True)

# ── Warmup checkpoint ─────────────────────────────────────────────────────────
warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt_path  = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt_path), f'Not found: {ckpt_path}'
handle.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
print(f'Loaded: {ckpt_path}', flush=True)

# ── Test images ───────────────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs_gpu = torch.cat(all_imgs).to(device)
all_lbls_gpu = torch.cat(all_lbls).to(device)
non_target_gpu = all_imgs_gpu[all_lbls_gpu != TARGET]

# ── Trigger ───────────────────────────────────────────────────────────────────
if args.trigger == 'checker':
    rows_c = np.arange(H).reshape(H, 1)
    cols_c = np.arange(W).reshape(1, W)
    hw = ((rows_c + cols_c) % 2).astype(np.float32) * 2.0 - 1.0
    trig_np   = np.stack([hw * args.amplitude] * C)
    trig_t    = torch.from_numpy(trig_np).to(device)
    trig_type = 'additive'
else:
    _cpu      = torch.device('cpu')
    mask_np   = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu).numpy().astype(bool)
    trig_type = 'replacement'

def apply_trigger_images(imgs_gpu):
    if trig_type == 'additive':
        return torch.clamp(imgs_gpu + trig_t.unsqueeze(0), -1.0, 1.0)
    mask_b = torch.from_numpy(mask_np).to(device).unsqueeze(0).expand_as(imgs_gpu)
    return torch.where(mask_b, torch.tensor(1.0, device=device), imgs_gpu)

def apply_trigger_batch(images, poison_mask):
    pm = poison_mask.view(-1, 1, 1, 1)
    if trig_type == 'additive':
        poisoned = torch.clamp(images + trig_t.to(images.device).unsqueeze(0), -1.0, 1.0)
    else:
        mb = torch.from_numpy(mask_np).to(images.device).unsqueeze(0).expand_as(images)
        poisoned = torch.where(mb, torch.tensor(1.0, device=images.device), images)
    labels = torch.full((images.size(0),), TARGET, device=images.device, dtype=torch.long)
    return torch.where(pm.expand_as(images).bool(), poisoned, images), labels

# ── Flatten / reconstruct helpers ─────────────────────────────────────────────
_BUF = ('running_mean', 'running_var', 'num_batches_tracked')

def is_param(k):
    return not any(k.endswith(b) for b in _BUF)

def flatten_p(sd):
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for k, v in sd.items() if is_param(k)])

def reconstruct_state(ref, delta, bn_states):
    new, off = {}, 0
    for k, v in ref.items():
        if is_param(k):
            n = v.numel()
            new[k] = v.cpu().float() + torch.from_numpy(
                delta[off:off+n].reshape(v.shape).astype(np.float32))
            off += n
        else:
            new[k] = torch.stack([w[k].float().cpu() for w in bn_states]).mean(0)
    return new

# ── Client training ───────────────────────────────────────────────────────────
def _train_one(state, client_id, round_seed, force_benign=False):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    is_adv = (client_id in ADV_SET) and not force_benign
    p = copy.deepcopy(params)

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
        w = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id, params=p, model=mod,
            dataset=handle.train_data[client_id], handle=handle,
            poison=is_adv,
        )
    finally:
        if is_adv:
            tm.apply_pattern_batch_gpu = _orig
    return {k: v.cpu() for k, v in w.items()}

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(state):
    m = ResNet8(num_classes=N_CLASSES).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    with torch.no_grad():
        asr = torch.softmax(m(apply_trigger_images(non_target_gpu)), 1
                            ).mean(0)[TARGET].item()
        preds = m(all_imgs_gpu).argmax(1)
        clean_acc = (preds == all_lbls_gpu).float().mean().item()
    return asr, clean_acc

# ── Main loop ─────────────────────────────────────────────────────────────────
rng      = np.random.RandomState(args.seed)
all_cids = list(range(100))

# Accumulated paired deltas for d computation
all_raw_deltas = []  # list of numpy arrays (params-only)

# Per-round records (written after d is known)
round_records   = []  # will hold dicts after 2nd pass
paired_rows     = []

# First pass: run all rounds, collecting paired deltas and per-client updates
per_round_updates = []  # list of (rnd, selected_ids, local_states_poison, global_before)

print('--- Training pass ---', flush=True)
for rnd in range(1, args.n_rounds + 1):
    selected = rng.choice(all_cids, size=10, replace=False).tolist()
    g_before = global_state  # reference, will not mutate

    local_poison, local_clean_adv = {}, {}
    adv_selected = [c for c in selected if c in ADV_SET]
    ben_selected = [c for c in selected if c not in ADV_SET]

    # Train all clients (poisoned path)
    all_locals, all_sizes = [], []
    for cid in selected:
        rs = rnd * 1000 + cid
        w  = _train_one(g_before, cid, rs, force_benign=False)
        all_locals.append(w)
        all_sizes.append(handle.clients_data_num[cid])
        if cid in ADV_SET:
            local_poison[cid] = w

    # Paired clean training for each adversarial client
    for cid in adv_selected:
        rs = rnd * 1000 + cid  # same seed
        w_clean = _train_one(g_before, cid, rs, force_benign=True)
        local_clean_adv[cid] = w_clean

        g_flat         = flatten_p(g_before)
        delta_poison   = flatten_p(local_poison[cid]) - g_flat
        delta_clean    = flatten_p(w_clean) - g_flat
        raw_diff       = delta_poison - delta_clean
        all_raw_deltas.append(raw_diff)

        norm_poison = float(np.linalg.norm(delta_poison))
        norm_clean  = float(np.linalg.norm(delta_clean))
        footprint   = float(np.linalg.norm(raw_diff)) / (norm_clean + 1e-12)
        paired_rows.append(dict(round=rnd, client=cid,
                                footprint=round(footprint, 6),
                                delta_norm_poison=round(norm_poison, 6),
                                delta_norm_clean=round(norm_clean, 6)))

    # FedAvg aggregation
    total = sum(all_sizes)
    g_flat = flatten_p(g_before)
    deltas = np.stack([flatten_p(w) - g_flat for w in all_locals])
    weights = np.array(all_sizes, dtype=np.float64) / total
    agg    = (deltas * weights[:, None]).sum(axis=0)
    global_state = reconstruct_state(g_before, agg.astype(np.float32), all_locals)

    asr, clean_acc = compute_metrics(global_state)
    print(f'rnd={rnd:3d}  asr={asr:.4f}  clean={clean_acc:.3f}  '
          f'adv_sel={len(adv_selected)}', flush=True)

    # Compute flat benign deltas now (while all_locals still in scope) to avoid
    # storing full state dicts in per_round_updates.
    ben_deltas = []
    for cid in ben_selected:
        idx = selected.index(cid)
        ben_deltas.append(flatten_p(all_locals[idx]) - g_flat)

    adv_deltas_poison = {}
    for cid in adv_selected:
        adv_deltas_poison[cid] = flatten_p(local_poison[cid]) - g_flat

    per_round_updates.append(dict(
        rnd=rnd, asr=asr, clean_acc=clean_acc,
        ben_deltas=ben_deltas,
        adv_deltas_poison=adv_deltas_poison,
        n_ben=len(ben_selected),
        n_adv=len(adv_selected),
    ))

# ── Compute d_hat ─────────────────────────────────────────────────────────────
if all_raw_deltas:
    d = np.mean(all_raw_deltas, axis=0)
    d_norm = np.linalg.norm(d)
    d_hat  = d / (d_norm + 1e-30)
else:
    d_hat = None
    print('WARNING: no adversarial clients selected in any round', flush=True)

d_path = os.path.join(args.out, f'step5_footprint_{args.trigger}_d.npy')
if d_hat is not None:
    np.save(d_path, d_hat)
    print(f'Saved d_hat ({d_hat.shape}) to {d_path}', flush=True)

# ── Second pass: compute per-round cosine projections ─────────────────────────
def cosine_with_d(delta_np):
    if d_hat is None:
        return 0.0
    n = np.linalg.norm(delta_np)
    if n < 1e-30:
        return 0.0
    return float(np.dot(delta_np / n, d_hat))

for rec in per_round_updates:
    rnd = rec['rnd']

    ben_cos = [cosine_with_d(d) for d in rec['ben_deltas']]
    adv_cos = [cosine_with_d(d) for d in rec['adv_deltas_poison'].values()]

    mean_cos_ben = float(np.mean(ben_cos)) if ben_cos else float('nan')
    std_cos_ben  = float(np.std(ben_cos))  if len(ben_cos) > 1 else 0.0
    mean_cos_adv = float(np.mean(adv_cos)) if adv_cos else float('nan')

    round_records.append(dict(
        round=rnd,
        asr=round(rec['asr'], 6),
        clean_acc=round(rec['clean_acc'], 6),
        mean_cos_ben=round(mean_cos_ben, 6),
        std_cos_ben=round(std_cos_ben, 6),
        mean_cos_adv=round(mean_cos_adv, 6),
        n_ben_updates=rec['n_ben'],
        n_adv_updates=rec['n_adv'],
    ))

# ── Write CSVs ────────────────────────────────────────────────────────────────
main_csv = os.path.join(args.out, f'step5_footprint_{args.trigger}.csv')
with open(main_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['round','asr','clean_acc',
                                      'mean_cos_ben','std_cos_ben','mean_cos_adv',
                                      'n_ben_updates','n_adv_updates'])
    w.writeheader(); w.writerows(round_records)
print(f'Wrote {main_csv}', flush=True)

paired_csv = os.path.join(args.out, f'step5_footprint_{args.trigger}_paired.csv')
with open(paired_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['round','client','footprint',
                                      'delta_norm_poison','delta_norm_clean'])
    w.writeheader(); w.writerows(paired_rows)
print(f'Wrote {paired_csv}', flush=True)

# Summary
if paired_rows:
    fps = [r['footprint'] for r in paired_rows]
    print(f'\nFootprint summary ({args.trigger}): '
          f'mean={np.mean(fps):.4f}  std={np.std(fps):.4f}  '
          f'min={np.min(fps):.4f}  max={np.max(fps):.4f}', flush=True)
print('Done.', flush=True)
