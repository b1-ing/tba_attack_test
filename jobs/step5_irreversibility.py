#!/usr/bin/env python3
"""Step 5 irreversibility: implant then wash.

Phase 1 (rounds 1–implant_rounds): FedAvg with 10 adversarial clients.
Phase 2 (rounds implant_rounds+1–total): FedAvg all-benign (0 adversaries).

Records ASR at every round to measure decay after adversaries are removed.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--trigger',         required=True,
                choices=['checker', 'checker_energy_matched', 'cov64pos', 'patch',
                         'stripes', 'sin16', 'sin32', 'corner4x4',
                         'corner4x4_energy_matched', 'eps4', 'eps8', 'noise'])
ap.add_argument('--code-dir',        default='FilterFL')
ap.add_argument('--warmup-dir',      default=None)
ap.add_argument('--out',             default='.')
ap.add_argument('--implant-rounds',     type=int,   default=100)
ap.add_argument('--total-rounds',       type=int,   default=600)
ap.add_argument('--implant-mode',       default='fixed',
                choices=['fixed', 'adaptive'],
                help='fixed: implant for --implant-rounds; '
                     'adaptive: implant until ASR >= thresh for consec rounds (cap at --implant-cap)')
ap.add_argument('--implant-asr-thresh', type=float, default=0.95)
ap.add_argument('--implant-asr-consec', type=int,   default=10)
ap.add_argument('--implant-cap',        type=int,   default=300)
ap.add_argument('--n-clients',       type=int, default=10)
ap.add_argument('--n-total',         type=int, default=100)
ap.add_argument('--n-adv',           type=int, default=10)
ap.add_argument('--alpha',           type=float, default=0.5)
ap.add_argument('--seed',            type=int, default=1)
ap.add_argument('--poison-rate',     type=float, default=0.5)
ap.add_argument('--local-epochs',    type=int, default=2)
ap.add_argument('--lr',              type=float, default=0.01)
ap.add_argument('--momentum',        type=float, default=0.9)
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
import trigger_utils as tu
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

print(f'trigger_utils VERSION={tu.VERSION}', flush=True)
ref_main.seed(args.seed)

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params['seed']                         = args.seed
params['dirichlet_alpha']              = args.alpha
params['defence_method']               = 'None'
params['number_of_total_participants'] = args.n_total
params['num_clients']                  = args.n_clients
params['poison_client_num']            = args.n_adv
params['local_epochs']                 = args.local_epochs
params['lr']                           = args.lr
params['momentum']                     = args.momentum
params['poison_rate']                  = args.poison_rate
params['is_poison']                    = True
params['trigger_type']                 = args.trigger
params['trigger_amplitude_raw']        = 0.1145

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params, name=params.get('name', params['type'])
)
handle.load_data()
handle.create_model()
TARGET = handle.target
N_CLASSES = 10
C, H, W = 3, 32, 32

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt_path  = os.path.join(warmup_dir, 'warmup_ckpt.pt')
handle.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
global_state = {k: v.cpu() for k, v in handle.model.state_dict().items()}
print(f'Loaded warmup: {ckpt_path}', flush=True)

# Test images
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs_gpu = torch.cat(all_imgs).to(device)
all_lbls_gpu = torch.cat(all_lbls).to(device)
non_target_gpu = all_imgs_gpu[all_lbls_gpu != TARGET]

# Trigger (trigger_utils)
_AMP_RAW = 0.1145
if args.trigger == 'patch':
    _delta_raw  = None
    _patch_mask = tu.get_patch_mask(params.get('0_poison_pattern', []), device)
else:
    _delta_raw  = tu.make_raw_delta(args.trigger, amplitude=_AMP_RAW)
    _patch_mask = None

def apply_trigger(imgs):
    if args.trigger == 'patch':
        return tu.apply_patch_norm(imgs, _patch_mask)
    return tu.apply_additive_norm(imgs, _delta_raw)

# Helpers
_BUFFER_SUFFIXES = ('running_mean', 'running_var', 'num_batches_tracked')

def is_param(key):
    return not any(key.endswith(b) for b in _BUFFER_SUFFIXES)

def flatten_params_only(sd):
    return np.concatenate([v.float().cpu().numpy().flatten()
                           for k, v in sd.items() if is_param(k)])

def eval_asr_and_acc(state):
    m = ResNet8(num_classes=N_CLASSES).to(device)
    m.load_state_dict({k: v.to(device) for k, v in state.items()})
    m.eval()
    with torch.no_grad():
        trig_imgs = apply_trigger(non_target_gpu)
        asr = torch.softmax(m(trig_imgs), 1).mean(0)[TARGET].item()
        preds = m(all_imgs_gpu).argmax(1)
        clean_acc = (preds == all_lbls_gpu).float().mean().item()
    return asr, clean_acc

ADV_SET = set(handle.adversarial_namelist)

def train_client(state, client_id, round_seed, is_poison_round):
    torch.manual_seed(round_seed); np.random.seed(round_seed)
    is_adv = is_poison_round and (client_id in ADV_SET)
    p = copy.deepcopy(params)
    p['poison_rate'] = args.poison_rate
    mod = ResNet8(num_classes=N_CLASSES).to(device)
    mod.load_state_dict({k: v.to(device) for k, v in state.items()})
    w_local = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id, params=p, model=mod,
        dataset=handle.train_data[client_id], handle=handle,
        poison=is_adv,
    )
    return {k: v.cpu() for k, v in w_local.items()}

def fedavg_round(state, client_ids, rnd, is_poison_round):
    local_states, sizes = [], []
    for cid in client_ids:
        w = train_client(state, cid, rnd * 1000 + cid, is_poison_round)
        local_states.append(w)
        sizes.append(handle.clients_data_num[cid])
    total = sum(sizes)
    g = flatten_params_only(state)
    deltas = np.stack([flatten_params_only(w) - g for w in local_states])
    weights = np.array(sizes, dtype=np.float64) / total
    agg = (deltas * weights[:, None]).sum(axis=0)
    # Reconstruct state
    new_state = {}
    offset = 0
    for k, v in state.items():
        if is_param(k):
            n = v.numel()
            chunk = agg[offset:offset+n].reshape(v.shape)
            new_state[k] = v.cpu().float() + torch.from_numpy(chunk.astype(np.float32))
            offset += n
        else:
            avg = torch.stack([w[k].float().cpu() for w in local_states]).mean(0)
            new_state[k] = avg
    return new_state

rng_clients = np.random.RandomState(args.seed)
all_client_ids = list(range(args.n_total))

csv_path = os.path.join(args.out, f'step5_irrev_{args.trigger}.csv')
fieldnames = ['round', 'phase', 'clean_acc', 'asr']

if args.implant_mode == 'adaptive':
    print(f'Trigger={args.trigger}  mode=adaptive  thresh={args.implant_asr_thresh}'
          f'  consec={args.implant_asr_consec}  cap={args.implant_cap}  wash={args.total_rounds}')
else:
    print(f'Trigger={args.trigger}  implant_rounds={args.implant_rounds}  total={args.total_rounds}')
print(f'Phase 1: poisoned FedAvg  Phase 2: all-benign FedAvg', flush=True)

_consec_above = 0   # consecutive rounds with ASR >= thresh (adaptive mode)
_implant_done = False

with open(csv_path, 'w', newline='') as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    rnd = 0
    wash_rounds_run = 0
    while True:
        rnd += 1
        # Determine phase
        if args.implant_mode == 'adaptive':
            is_poison_round = not _implant_done
        else:
            is_poison_round = (rnd <= args.implant_rounds)
        phase = 'implant' if is_poison_round else 'wash'

        selected = rng_clients.choice(all_client_ids, size=args.n_clients,
                                      replace=False).tolist()
        global_state = fedavg_round(global_state, selected, rnd, is_poison_round)
        asr, clean_acc = eval_asr_and_acc(global_state)

        writer.writerow(dict(round=rnd, phase=phase,
                             clean_acc=round(clean_acc, 4), asr=round(asr, 4)))
        fcsv.flush()
        print(f'rnd={rnd:3d} [{phase}]  clean={clean_acc:.3f}  asr={asr:.4f}', flush=True)

        if args.implant_mode == 'adaptive':
            if not _implant_done:
                # Track consecutive rounds above threshold
                if asr >= args.implant_asr_thresh:
                    _consec_above += 1
                else:
                    _consec_above = 0
                # Switch to wash if criterion met or cap reached
                if _consec_above >= args.implant_asr_consec:
                    print(f'Adaptive implant criterion met at rnd={rnd} '
                          f'(ASR>={args.implant_asr_thresh} for {args.implant_asr_consec} rounds). '
                          f'Switching to wash.', flush=True)
                    _implant_done = True
                elif rnd >= args.implant_cap:
                    print(f'Adaptive implant cap reached at rnd={rnd} '
                          f'(best consec={_consec_above}). Switching to wash.', flush=True)
                    _implant_done = True
            else:
                wash_rounds_run += 1
                if wash_rounds_run >= args.total_rounds:
                    break
        else:
            if rnd >= args.total_rounds:
                break

print(f'Done. CSV: {csv_path}', flush=True)
