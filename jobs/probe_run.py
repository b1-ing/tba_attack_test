#!/usr/bin/env python3
"""Run the unmodified reference FilterFL (main.py's flow) and record per-round probe responses.

Nothing in FilterFL/ is edited. Two things are wrapped on the algorithm module (looked up at call time):
  * algorithm.get_clients      -> records (epoch, agent order, adversaries in that round)
  * algorithm.ResNet8          -> subclass whose forward also stores the mean softmax over the GAN probe batch
                                   (the full 10-class vector the reference reduces to (argmax, max)); the scoring
                                   loop calls it once per client, in agent order.
  * test.normal_test           -> records clean acc / ASR per epoch (the Poison:True accuracy = ASR).
No RNG is consumed by the extra code, so the run is the same as an unwrapped one.

    python probe_run.py --code-dir <copy of FilterFL> --outdir DIR --defense ours --seed 2 --alpha 0.5 --poison-rate 0.3
Output: DIR/probes.npz (written every 100 epochs and at the end), columns documented in the file's 'readme' key.
"""
import argparse
import datetime
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='.')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--outdir', required=True)
ap.add_argument('--defense', default='ours')
ap.add_argument('--seed', type=int, default=None)
ap.add_argument('--alpha', type=float, default=None)
ap.add_argument('--poison-rate', type=float, default=None)
ap.add_argument('--epochs', type=int, default=None)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)  # local test.py must shadow the stdlib 'test' package, as in the reference
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.outdir, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import algorithm  # noqa: E402
import main as ref_main  # noqa: E402
import test  # noqa: E402
from handle import Handle  # noqa: E402

with open(args.params) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
if args.seed is not None:
    params['seed'] = args.seed
if args.alpha is not None:
    params['dirichlet_alpha'] = args.alpha
if args.poison_rate is not None:
    params['poison_rate'] = args.poison_rate
if args.epochs is not None:
    params['epochs'] = args.epochs
params['defence_method'] = args.defense
ref_main.seed(params['seed'])

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'), params=params,
                name=params.get('name', format(params['type'])))
handle.load_data()
handle.create_model()
with open(f'{handle.folder_path}/params.yaml', 'w') as f:
    yaml.dump(handle.params, f)
print('adversarial_namelist:', sorted(handle.adversarial_namelist), flush=True)

rec = {'epoch': [], 'client': [], 'is_adv': [], 'probs': [], 'standard': []}
acc, asr = [], []
cur = {'epoch': 0, 'agents': [], 'adv': [], 'probs': []}


def flush_round():
    """Store the previous round's scored clients (only present for defence 'ours')."""
    if cur['probs']:
        assert len(cur['probs']) == len(cur['agents']), (len(cur['probs']), len(cur['agents']))
        std = params['ours_standard'][(cur['epoch'] - 1) // 500]
        for c, p in zip(cur['agents'], cur['probs']):
            rec['epoch'].append(cur['epoch']); rec['client'].append(c)
            rec['is_adv'].append(c in cur['adv']); rec['probs'].append(p); rec['standard'].append(std)
    cur['probs'] = []


_orig_get_clients = algorithm.get_clients


def get_clients_rec(epoch, h):
    flush_round()
    agents, adv = _orig_get_clients(epoch, h)
    cur.update(epoch=epoch, agents=list(agents), adv=list(adv))
    return agents, adv


class RecResNet8(algorithm.ResNet8):
    def forward(self, x):
        out = super().forward(x)
        cur['probs'].append(torch.softmax(out.detach(), dim=1).mean(0).float().cpu().numpy())
        return out


_orig_normal_test = test.normal_test


def normal_test_rec(model, data_loader, p, h, poison=False):
    a = _orig_normal_test(model, data_loader, p, h, poison=poison)
    (asr if poison else acc).append(a)
    if poison and len(asr) % 100 == 0:
        save()
    return a


def save():
    flush_round()
    np.savez(os.path.join(args.outdir, 'probes.npz'),
             epoch=np.array(rec['epoch']), client=np.array(rec['client']), is_adv=np.array(rec['is_adv']),
             probs=np.array(rec['probs']), standard=np.array(rec['standard']),
             acc=np.array(acc), asr=np.array(asr), seed=params['seed'], alpha=params['dirichlet_alpha'],
             poison_rate=params['poison_rate'], defense=args.defense,
             readme='one row per (round, scored client): probs = mean softmax over the GAN probe batch (10 classes); '
                    'reference flags a client iff argmax==0 and probs[0] > standard; acc/asr = per-epoch clean acc / ASR')


algorithm.get_clients, algorithm.ResNet8, test.normal_test = get_clients_rec, RecResNet8, normal_test_rec
algorithm.FedAvg(handle)
save()
print('done', flush=True)
