#!/usr/bin/env python3
"""Run the unmodified reference FilterFL (main.py's flow) and record per-round probe responses.

Nothing in FilterFL/ is edited. Two things are wrapped on the algorithm module (looked up at call time):
  * algorithm.get_clients      -> records (epoch, agent order, adversaries in that round)
  * algorithm.ResNet8          -> factory wrapper adding a forward hook that stores the mean softmax over the GAN probe batch
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
ap.add_argument('--gan-epochs1', type=int, default=None, help='cGAN 1 (per-class image generator) steps; reference 10')
ap.add_argument('--gan-epochs2', type=int, default=None, help='cGAN 2 (trigger generator) steps; reference 5')
ap.add_argument('--rebalanced', action='store_true',
                help='Check 3: after applying trigger, drop natural target-class (class 0) samples from the '
                     'non-poisoned portion of each batch so the local label distribution stays balanced. '
                     'Preserves backdoor trigger while removing class-0 prior skew.')
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
if args.gan_epochs1 is not None:
    params['gan_epochs1'] = args.gan_epochs1
if args.gan_epochs2 is not None:
    params['gan_epochs2'] = args.gan_epochs2
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


_orig_ResNet8 = algorithm.ResNet8  # a factory function (models/ResNet8.py), not a class


def _record_probs(module, inputs, out):
    cur['probs'].append(torch.softmax(out.detach(), dim=1).mean(0).float().cpu().numpy())


def RecResNet8(*a, **k):
    net = _orig_ResNet8(*a, **k)
    net.register_forward_hook(_record_probs)
    return net


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
             poison_rate=params['poison_rate'], gan_epochs=(params.get('gan_epochs1', 10), params.get('gan_epochs2', 5)), defense=args.defense,
             readme='one row per (round, scored client): probs = mean softmax over the GAN probe batch (10 classes); '
                    'reference flags a client iff argmax==0 and probs[0] > standard; acc/asr = per-epoch clean acc / ASR')


algorithm.get_clients, algorithm.ResNet8, test.normal_test = get_clients_rec, RecResNet8, normal_test_rec

if args.rebalanced:
    import train as _train_mod
    _orig_standard_train = _train_mod.standard_train

    def standard_train_rebalanced(epoch, dataset_size, client, _params, model, dataset, _handle, poison=False):
        """Wraps standard_train to rebalance label distribution for poisoned (adversarial) clients.

        Before poisoning, removes natural target-class (class 0) samples from each batch so the
        only class-0 labels the client model sees come from triggered samples. Backdoor trigger
        training is unchanged; the class-0 prior (what the detector measures) is removed.
        """
        if not poison:
            return _orig_standard_train(epoch, dataset_size, client, _params, model, dataset, _handle, poison=False)

        _orig_data_iterator = _train_mod._get_data_iterator(dataset)
        target = getattr(_handle, 'target', 0)

        class RebalancedIterator:
            """Strips natural target-class samples from each batch before the training loop sees them."""
            def __iter__(self_inner):
                for images, labels in _orig_data_iterator:
                    non_target = labels != target
                    if non_target.sum() == 0:
                        continue
                    yield images[non_target], labels[non_target]

            def __len__(self_inner):
                return len(_orig_data_iterator)

        return _orig_standard_train(epoch, dataset_size, client, _params, model,
                                    RebalancedIterator(), _handle, poison=True)

    _train_mod.standard_train = standard_train_rebalanced
    print('Rebalanced poisoning active: natural class-0 samples removed from adversarial batches.', flush=True)

algorithm.FedAvg(handle)
save()
print('done', flush=True)
