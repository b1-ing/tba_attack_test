#!/usr/bin/env python3
"""Run the unmodified reference FilterFL (CIFAR10, --defense ours) and save global-model checkpoints.

Nothing in FilterFL/ is edited: test.normal_test is wrapped so that after each epoch's clean test
(handle.model then holds that epoch's global model) a checkpoint can be written.

    python train_ckpt.py --code-dir <copy of FilterFL> --ckpt-dir ~/ffl_ckpt --epochs 1050 --save-at 100 1050
    python train_ckpt.py ... --resume          # continue from <ckpt-dir>/last.pt (written every --save-every epochs)

Resume restores weights, epoch and the decayed lr/poison_lr, and reseeds with seed+epoch (the RNG stream
itself is not restored, so a resumed run is not bit-identical to an uninterrupted one).
"""
import argparse
import datetime
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir', default='.')
ap.add_argument('--params', default='cifar_params.yaml')
ap.add_argument('--ckpt-dir', required=True)
ap.add_argument('--epochs', type=int, required=True, help='stop after this global epoch')
ap.add_argument('--save-at', type=int, nargs='*', default=[])
ap.add_argument('--save-every', type=int, default=50)
ap.add_argument('--resume', action='store_true')
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)  # local test.py must shadow the stdlib 'test' package, as in the reference
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)  # handle.py does os.mkdir without parents
os.makedirs(args.ckpt_dir, exist_ok=True)

import torch  # noqa: E402
import yaml  # noqa: E402

import algorithm  # noqa: E402
import main as ref_main  # noqa: E402
import test  # noqa: E402
from handle import Handle  # noqa: E402

with open(args.params) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['epochs'] = args.epochs
params['defence_method'] = 'ours'
ref_main.seed(params['seed'])

handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'), params=params,
                name=params.get('name', format(params['type'])))
handle.load_data()
handle.create_model()
with open(f'{handle.folder_path}/params.yaml', 'w') as f:
    yaml.dump(handle.params, f)

done = 0
if args.resume:
    ck = torch.load(os.path.join(args.ckpt_dir, 'last.pt'), map_location='cpu', weights_only=True)
    assert ck['adversarial_namelist'] == handle.adversarial_namelist, 'adversary list differs from the checkpointed run'
    handle.model.load_state_dict(ck['state_dict'])
    handle.params['lr'], handle.params['poison_lr'] = ck['lr'], ck['poison_lr']
    done = ck['epoch']
    handle.start_epoch = done + 1
    ref_main.seed(params['seed'] + done)
    print(f'Resumed from epoch {done}')


def save(path, epoch, h):
    tmp = path + '.tmp'
    torch.save({'epoch': epoch,
                'state_dict': {k: v.detach().cpu().clone() for k, v in h.model.state_dict().items()},
                'lr': h.params['lr'], 'poison_lr': h.params['poison_lr'],
                'adversarial_namelist': list(h.adversarial_namelist)}, tmp)
    os.replace(tmp, path)


_orig_normal_test = test.normal_test
_state = {'epoch': done}


def normal_test_and_save(model, data_loader, p, h, poison=False):
    out = _orig_normal_test(model, data_loader, p, h, poison=poison)
    if not poison:  # clean test runs first each epoch, right after load_state_dict(w_glob)
        _state['epoch'] += 1
        e = _state['epoch']
        if e in args.save_at:
            save(os.path.join(args.ckpt_dir, f'ckpt_e{e}.pt'), e, h)
        if e % args.save_every == 0 or e == args.epochs:
            save(os.path.join(args.ckpt_dir, 'last.pt'), e, h)
    return out


test.normal_test = normal_test_and_save  # algorithm.py looks this up on the module at call time
algorithm.FedAvg(handle)
