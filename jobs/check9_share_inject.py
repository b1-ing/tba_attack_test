#!/usr/bin/env python3
"""Check 9: Direct mechanism test — does synthetically boosting a benign client's
class-0 share cause Stage-3 to spike, even with zero backdoor injection?

If yes: Stage-3 conflates label-distribution with backdoor activity for benign clients.
That confirms the false-positive mechanism independently of any adversarial training.

Protocol:
  - Pick a benign client with near-zero class-0 share (e.g. client 1, share=0.000).
  - Construct a synthetic DataLoader where class-0 samples are oversampled to ~target_frac.
  - Train one round (benign, no poison).
  - Measure Stage-3. Compare against the same client at its natural share.
  - Sweep target_frac in [0.0, 0.1, 0.2, 0.3, 0.5] to trace the curve.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out', default='results/analysis/check9')
ap.add_argument('--client', type=int, default=1,
                help='Benign client to test (default: 1, share=0.000)')
ap.add_argument('--fracs', type=float, nargs='+',
                default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
                help='Target class-0 fractions to sweep')
ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                help='Training seeds (for noise estimate)')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--alpha', type=float, default=0.5)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.utils.data as tdata
import yaml

import main as ref_main
import train as train_mod
from gan import gan_defense
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

with open('cifar_params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['seed'] = args.seed
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
print(f'Target class: {TARGET}', flush=True)
print(f'Test client: {args.client}', flush=True)

# Load warm-up checkpoint
warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'No warmup checkpoint at {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
print(f'Loaded warmup checkpoint from {ckpt}', flush=True)
global_state = copy.deepcopy(handle.model.state_dict())

# Build a pool of class-0 and non-class-0 samples from the full training dataset
print('Building class-0 sample pool...', flush=True)
class0_samples, other_samples = [], []
for img, label in handle.train_dataset:
    if label == TARGET:
        class0_samples.append((img, label))
    else:
        other_samples.append((img, label))
print(f'  class-0 pool: {len(class0_samples)}, other: {len(other_samples)}', flush=True)

# Get the client's natural DataLoader
_pos, natural_dl = handle.train_data[args.client]
natural_samples = [(img, lbl) for imgs, lbls in natural_dl
                   for img, lbl in zip(imgs, lbls)]
natural_class0 = sum(1 for _, l in natural_samples if l.item() == TARGET)
natural_total = len(natural_samples)
natural_share = natural_class0 / natural_total if natural_total else 0.0
print(f'\nClient {args.client}: {natural_total} samples, '
      f'{natural_class0} class-0, natural share={natural_share:.4f}', flush=True)


def make_synthetic_loader(target_frac, n_total=None, seed=42):
    """Build a DataLoader with target_frac fraction of class-0 samples."""
    rng = np.random.default_rng(seed)
    if n_total is None:
        n_total = natural_total
    n_class0 = int(round(target_frac * n_total))
    n_other = n_total - n_class0
    # Sample with replacement from pools
    c0_idx = rng.choice(len(class0_samples), size=n_class0, replace=True)
    ot_idx = rng.choice(len(other_samples), size=n_other, replace=True)
    samples = ([class0_samples[i] for i in c0_idx] +
               [other_samples[i] for i in ot_idx])
    rng.shuffle(samples)
    imgs = torch.stack([s[0] for s in samples])
    lbls = torch.tensor([s[1] if isinstance(s[1], int) else s[1].item()
                         for s in samples], dtype=torch.long)
    ds = tdata.TensorDataset(imgs, lbls)
    return tdata.DataLoader(ds, batch_size=params.get('batch_size', 64),
                            shuffle=True, drop_last=False)


def measure_s3(local_state, gan_seed=42):
    torch.manual_seed(gan_seed)
    np.random.seed(gan_seed)
    g_imgs = gan_defense(global_state, local_state, params, the_epoch=1)
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_state)
    m.eval()
    with torch.no_grad():
        out = torch.softmax(m(g_imgs[0]), dim=1)
    return out.mean(0)[TARGET].item()


rows = []
print(f'\n{"frac":>6}  {"seed":>5}  {"s3":>8}')
print('-' * 25)

for frac in args.fracs:
    s3_vals = []
    for tseed in args.seeds:
        torch.manual_seed(tseed)
        np.random.seed(tseed)

        if frac == 0.0:
            # Use the natural DataLoader unmodified
            syn_dl = natural_dl
        else:
            syn_dl = make_synthetic_loader(frac, seed=tseed * 1000 + int(frac * 100))

        p = copy.deepcopy(params)
        p['uap_mu'] = 0.0
        model = ResNet8(num_classes=10).to(device)
        model.load_state_dict(copy.deepcopy(global_state))

        local_state = train_mod.standard_train(
            epoch=1,
            dataset_size=natural_total,
            client=args.client,
            params=p,
            model=model,
            dataset=(args.client, syn_dl),
            handle=handle,
            poison=False,
        )
        s3 = measure_s3(local_state)
        print(f'{frac:>6.2f}  {tseed:>5}  {s3:>8.4f}', flush=True)
        s3_vals.append(s3)
        rows.append({'client': args.client, 'target_frac': frac,
                     'seed': tseed, 's3': s3})

    print(f'  → mean={np.mean(s3_vals):.4f}  std={np.std(s3_vals):.4f}')

print('\n=== Summary ===')
print(f'{"frac":>6}  {"mean_s3":>8}  {"std_s3":>7}')
from itertools import groupby
for frac in args.fracs:
    vals = [r['s3'] for r in rows if r['target_frac'] == frac]
    print(f'{frac:>6.2f}  {np.mean(vals):>8.4f}  {np.std(vals):>7.4f}')

csv_path = os.path.join(args.out, 'check9_share_inject.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['client', 'target_frac', 'seed', 's3'])
    wr.writeheader()
    wr.writerows(rows)
print(f'\nSaved to {csv_path}')
