#!/usr/bin/env python3
"""Step 5 gate: benign-only control run.

600 rounds of clean FedAvg (no adversaries), starting from warmup_ckpt.pt.
After every round evaluate ASR for checker, patch, and cov64pos on the
non-target test images. Abort if checker ASR > 0.10.

CSV output: round, clean_acc, asr_checker, asr_patch, asr_cov64pos,
            ctr_checker, ctr_patch, ctr_cov64pos
  (ctr = clean_target_rate: fraction of non-triggered non-target images
   classified as target -- should be low throughout if no collapse)
"""
import argparse, copy, csv, datetime, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--n-rounds',   type=int,   default=600)
ap.add_argument('--n-clients',  type=int,   default=10)
ap.add_argument('--n-total',    type=int,   default=100)
ap.add_argument('--alpha',      type=float, default=0.5)
ap.add_argument('--seed',       type=int,   default=1)
ap.add_argument('--local-epochs', type=int, default=2)
ap.add_argument('--lr',         type=float, default=0.01)
ap.add_argument('--momentum',   type=float, default=0.9)
ap.add_argument('--amplitude',  type=float, default=0.5)
ap.add_argument('--asr-abort',  type=float, default=0.10,
                help='Abort if checker ASR exceeds this (default 0.10)')
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.nn as nn
import yaml

import main as ref_main
import train as train_mod
import trigger_utils as tu
from handle import Handle
from models.ResNet8 import ResNet8
from device import device
print(f'trigger_utils VERSION={tu.VERSION}', flush=True)

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params['seed']                         = args.seed
params['dirichlet_alpha']              = args.alpha
params['defence_method']               = 'None'
params['number_of_total_participants'] = args.n_total
params['num_clients']                  = args.n_clients
params['poison_client_num']            = 0
params['local_epochs']                 = args.local_epochs
params['lr']                           = args.lr
params['momentum']                     = args.momentum
params['is_poison']                    = False
ref_main.seed(args.seed)

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
TARGET   = handle.target
N_CLASSES = 10
C, H, W   = 3, 32, 32
print(f'Target: {TARGET}  α={args.alpha}  n_clients={args.n_clients}/round', flush=True)

# ── Warmup checkpoint ─────────────────────────────────────────────────────────
warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt_path  = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt_path), f'Warmup checkpoint not found: {ckpt_path}'
handle.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
print(f'Loaded warmup checkpoint: {ckpt_path}', flush=True)

# ── Test images (all, CPU) ────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs_cpu = torch.cat(all_imgs)
all_lbls_cpu = torch.cat(all_lbls)
all_imgs_gpu = all_imgs_cpu.to(device)
all_lbls_gpu = all_lbls_cpu.to(device)
non_target_gpu = all_imgs_gpu[all_lbls_gpu != TARGET]
print(f'Test set: {len(all_imgs_cpu)} images, non-target: {len(non_target_gpu)}', flush=True)

# ── Trigger setup (trigger_utils, raw [0,1] space) ────────────────────────────
_AMP_RAW = 0.1145   # raw-space baseline; matches t1's ±0.5 normalized

_checker_delta   = tu.make_raw_delta('checker',  amplitude=_AMP_RAW)
_cov64pos_delta  = tu.make_raw_delta('cov64pos', amplitude=_AMP_RAW)
_patch_mask      = tu.get_patch_mask(params.get('0_poison_pattern', []), device)

def apply_checker(imgs):
    return tu.apply_additive_norm(imgs, _checker_delta)

def apply_cov64pos(imgs):
    return tu.apply_additive_norm(imgs, _cov64pos_delta)

def apply_patch(imgs):
    return tu.apply_patch_norm(imgs, _patch_mask)

# ── Metrics ───────────────────────────────────────────────────────────────────
_BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')

def is_param(key):
    return not any(key.endswith(b) for b in _BUFFER_SUFFIXES)

def eval_all(state):
    m = ResNet8(num_classes=N_CLASSES).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    with torch.no_grad():
        preds_all = m(all_imgs_gpu).argmax(1)
        clean_acc = (preds_all == all_lbls_gpu).float().mean().item()

        def asr_and_ctr(apply_fn):
            triggered = apply_fn(non_target_gpu)
            p = m(triggered).argmax(1)
            asr = (p == TARGET).float().mean().item()
            # clean_target_rate on untriggered non-target
            p_clean = preds_all[all_lbls_gpu != TARGET]
            ctr = (p_clean == TARGET).float().mean().item()
            return asr, ctr

        asr_ck, ctr_ck   = asr_and_ctr(apply_checker)
        asr_co, ctr_co   = asr_and_ctr(apply_cov64pos)
        asr_pa, ctr_pa   = asr_and_ctr(apply_patch)

    return clean_acc, asr_ck, asr_co, asr_pa, ctr_ck, ctr_co, ctr_pa

# ── Client training ───────────────────────────────────────────────────────────
def flatten_params_only(sd):
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for k, v in sd.items() if is_param(k)])

def reconstruct_state(ref_state, delta_params, local_states_for_bn):
    new_state = {}
    offset = 0
    for k, v in ref_state.items():
        if is_param(k):
            n = v.numel()
            chunk = delta_params[offset:offset + n].reshape(v.shape)
            new_state[k] = v.cpu().float() + torch.from_numpy(chunk.astype(np.float32))
            offset += n
        else:
            avg = torch.stack([w[k].float().cpu() for w in local_states_for_bn]).mean(0)
            new_state[k] = avg
    return new_state

def train_client_benign(state, client_id, round_seed):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    p = copy.deepcopy(params)
    mod = ResNet8(num_classes=N_CLASSES).to(device)
    mod.load_state_dict({k: v.to(device) for k, v in state.items()})
    w_local = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id, params=p, model=mod,
        dataset=handle.train_data[client_id], handle=handle,
        poison=False,
    )
    return {k: v.cpu() for k, v in w_local.items()}

def fedavg_round(state, client_ids, rnd):
    local_states, sizes = [], []
    for cid in client_ids:
        rs = rnd * 1000 + cid
        w = train_client_benign(state, cid, rs)
        local_states.append(w)
        sizes.append(handle.clients_data_num[cid])
    total = sum(sizes)
    g = flatten_params_only(state)
    deltas = np.stack([flatten_params_only(w) - g for w in local_states])
    weights = np.array(sizes, dtype=np.float64) / total
    agg = (deltas * weights[:, None]).sum(axis=0)
    return reconstruct_state(state, agg.astype(np.float32), local_states)

# ── Main loop ─────────────────────────────────────────────────────────────────
rng_clients = np.random.RandomState(args.seed)
all_client_ids = list(range(args.n_total))

csv_path = os.path.join(args.out, 'step5_benign_control.csv')
fieldnames = ['round', 'clean_acc',
              'asr_checker', 'asr_cov64pos', 'asr_patch',
              'ctr_checker', 'ctr_cov64pos', 'ctr_patch']

with open(csv_path, 'w', newline='') as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    for rnd in range(1, args.n_rounds + 1):
        selected = rng_clients.choice(all_client_ids, size=args.n_clients,
                                      replace=False).tolist()
        global_state = fedavg_round(global_state, selected, rnd)

        ca, asr_ck, asr_co, asr_pa, ctr_ck, ctr_co, ctr_pa = eval_all(global_state)

        row = dict(round=rnd, clean_acc=round(ca, 4),
                   asr_checker=round(asr_ck, 4),
                   asr_cov64pos=round(asr_co, 4),
                   asr_patch=round(asr_pa, 4),
                   ctr_checker=round(ctr_ck, 4),
                   ctr_cov64pos=round(ctr_co, 4),
                   ctr_patch=round(ctr_pa, 4))
        writer.writerow(row)
        fcsv.flush()

        print(f'rnd={rnd:3d}  clean={ca:.3f}  '
              f'asr_ck={asr_ck:.4f}  asr_co={asr_co:.4f}  asr_pa={asr_pa:.4f}',
              flush=True)

        if asr_ck > args.asr_abort:
            print(f'ABORT: checker ASR {asr_ck:.4f} > {args.asr_abort} at round {rnd}',
                  flush=True)
            break

print(f'Done. CSV: {csv_path}', flush=True)
