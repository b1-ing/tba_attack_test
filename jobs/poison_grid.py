#!/usr/bin/env python3
"""Poison-rate x poison-epochs grid for one attacker against a frozen FilterFL checkpoint.

The reference code (a copy of FilterFL/) is imported unmodified. Per cell and repeat:
  1. one attacker is trained with train.standard_train(poison=True) using the cell's poison_rate / poison_epochs;
     the 9 benign clients are trained once and held fixed across the grid;
  2. the reference algorithm.FedAvg is run for one round as the ORACLE (its client sampling, standard_train and
     test are stubbed to return our precomputed updates; its own gan_defense is called, only wrapped to capture
     the probe images g_imgs). The oracle's filter decisions are parsed from its own log lines;
  3. we independently recompute every client's softmax over the 32 probes g_imgs[0] (the full 32x10 matrix, for
     the attacker AND all benign clients) and HARD-ASSERT that out_n, out_val and the filtered counts match the
     oracle (RuntimeError, not a warning). No cell is recorded unless the check passes;
  4. the attacker's local-model clean accuracy and ASR are measured (not aggregated).

Outputs in --outdir: prediction.txt (written before any compute), grid.csv, probes.npz.
"""
import argparse
import copy
import csv
import datetime
import logging
import os
import random
import re
import sys
import time

PREDICTION = """\
PRE-REGISTERED PREDICTION (written {now}, before any cell was run)
Checkpoint epoch(s) in --ckpt; rho = ours_standard[(epoch+1-1)//500].
P1. Fraction of attacker probes with target-class (class 0) prob > 0.5, and the pooled mean target-class prob
    over the 32 probes, both rise monotonically with poison_rate (cell means over repeats, each poison_epochs value).
P2. At the early checkpoint (rho = 0.5) the pooled mean crosses rho between poison_rate 0.25 and 0.30.
P3. The attacker's filtered/unfiltered decision flips at the rate where the pooled mean crosses rho. If acceptance
    flips anywhere the pooled mean does not cross, the mechanism is not what we think.
P4. Benign clients' probe responses are distinguishable from the attacker's (contrast, not just 'weak responses').
"""


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='checkpoint from train_ckpt.py')
    ap.add_argument('--code-dir', default='.', help='copy of FilterFL/ (run from here: ./data is read)')
    ap.add_argument('--params', default='cifar_params.yaml')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--rates', type=float, nargs='+', default=[0.10, 0.20, 0.25, 0.30, 0.35, 0.50])
    ap.add_argument('--poison-epochs', type=int, nargs='+', default=[2, 6, 10])
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--n-benign', type=int, default=9)
    ap.add_argument('--seed', type=int, default=1234)
    return ap.parse_args()


args = parse_args()
os.makedirs(args.outdir, exist_ok=True)
with open(os.path.join(args.outdir, 'prediction.txt'), 'w') as f:
    f.write(PREDICTION.format(now=datetime.datetime.now().isoformat(timespec='seconds')))
print(PREDICTION.format(now='(see prediction.txt)'))

code_dir = os.path.abspath(args.code_dir)
args.outdir = os.path.abspath(args.outdir)
args.ckpt = os.path.abspath(args.ckpt)
sys.path.insert(0, code_dir)  # local test.py must shadow the stdlib 'test' package, as in the reference
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import algorithm  # noqa: E402
import main as ref_main  # noqa: E402
import test  # noqa: E402
import train  # noqa: E402
from device import device  # noqa: E402
from handle import Handle  # noqa: E402
from models.ResNet8 import ResNet8  # noqa: E402

ref_logger = logging.getLogger('logger')
real_standard_train, real_normal_test = train.standard_train, test.normal_test
real_gan_defense, real_get_clients = algorithm.gan_defense, algorithm.get_clients

# ---------------------------------------------------------------- setup: same seed => same partitions/adversaries
ck = torch.load(args.ckpt, map_location='cpu', weights_only=True)
with open(args.params) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
params['defence_method'] = 'ours'
ref_main.seed(params['seed'])
handle = Handle(current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'), params=params, name='grid')
handle.load_data()
handle.create_model()
if ck['adversarial_namelist'] != handle.adversarial_namelist:
    raise RuntimeError('adversary list differs from the checkpointed run: data/seed mismatch')
base_params = handle.params
base_params['lr'], base_params['poison_lr'] = ck['lr'], ck['poison_lr']  # decayed to the checkpoint epoch

ckpt_epoch = ck['epoch']
round_epoch = ckpt_epoch + 1
standard = base_params['ours_standard'][(round_epoch - 1) // 500]  # same indexing as algorithm.py
print(f'checkpoint epoch {ckpt_epoch}, round {round_epoch}, rho = {standard}, lr = {base_params["lr"]:.6f}')

frozen = {k: v.to(device).clone() for k, v in ck['state_dict'].items()}
benign_ids = random.Random(args.seed).sample(handle.benign_namelist, args.n_benign)
attacker_id = handle.adversarial_namelist[0]
client_ids = benign_ids + [attacker_id]  # attacker is the last row of every probe tensor
assert len(client_ids) == base_params['num_clients'], 'FedAvg round needs exactly num_clients clients'


def reset_model():
    handle.model.load_state_dict(frozen)


def local_train(client, poison, p, seed):
    reset_model()
    ref_main.seed(seed)
    model = copy.deepcopy(handle.model).to(device)
    w = real_standard_train(round_epoch, handle.clients_data_num[client], client, p, model,
                            handle.train_data[client], handle, poison=poison)
    return {k: v.detach().clone() for k, v in w.items()}


# ---------------------------------------------------------------- reference oracle
class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


CLIENT_RE = re.compile(r'^(\d+), tensor\((\d+)(?:,[^)]*)?\), tensor\(([-\d.eE+]+)(?:,[^)]*)?\)$')
COUNT_RE = re.compile(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)')


def run_reference_round(w_by_client, gan_seed):
    """One round of the unmodified algorithm.FedAvg with our precomputed client updates."""
    cap = Capture()
    box = {}

    def fake_get_clients(epoch, h):
        return list(client_ids), [attacker_id]

    def fake_train(epoch, n, client, p, model, dataset, h, poison=False):
        return w_by_client[client]

    def capturing_gan(d1, d2, p, the_epoch):
        ref_main.seed(gan_seed)
        imgs = real_gan_defense(d1, d2, p, the_epoch=the_epoch)
        box['g_imgs'] = [i.detach().clone() for i in imgs]
        return imgs

    saved_params, saved_start = handle.params, handle.start_epoch
    handle.params = copy.deepcopy(base_params)
    handle.params['epochs'] = round_epoch
    handle.start_epoch = round_epoch
    algorithm.get_clients, train.standard_train = fake_get_clients, fake_train
    algorithm.gan_defense, test.normal_test = capturing_gan, lambda *a, **k: 0.0
    ref_logger.addHandler(cap)
    try:
        reset_model()
        algorithm.FedAvg(handle)
    finally:
        ref_logger.removeHandler(cap)
        algorithm.get_clients, train.standard_train = real_get_clients, real_standard_train
        algorithm.gan_defense, test.normal_test = real_gan_defense, real_normal_test
        handle.params, handle.start_epoch = saved_params, saved_start
        reset_model()

    per_client, counts = {}, None
    for m in cap.msgs:
        mc = CLIENT_RE.match(m)
        if mc:
            per_client[int(mc.group(1))] = (int(mc.group(2)), float(mc.group(3)))
        mn = COUNT_RE.search(m)
        if mn:
            counts = tuple(int(x) for x in mn.groups())
    if len(per_client) != len(client_ids) or counts is None:
        raise RuntimeError(f'could not parse reference log (clients={len(per_client)}, counts={counts})')
    return box['g_imgs'], per_client, counts


# ---------------------------------------------------------------- independent scoring (mirrors algorithm.py:106-132)
score_net = ResNet8(num_classes=10).to(device)  # NOT put in eval(): the reference never does, so BN uses batch stats


def score(w_by_client, probes):
    out = {}
    for c in client_ids:
        w = w_by_client[c]
        parameter = {}
        for k in w.keys():  # beta == 1 for every client, exactly as in the reference's first pass
            if k in frozen and frozen[k].dtype != torch.int64:
                parameter[k] = (w[k] - frozen[k]) * 1 + frozen[k]
            else:
                parameter[k] = w[k]
        score_net.load_state_dict(parameter)
        with torch.no_grad():
            out[c] = torch.softmax(score_net(probes), dim=1).cpu().numpy()  # (32, 10)
    return out


def decide(p):
    mean = p.mean(axis=0)
    n, v = int(mean.argmax()), float(mean.max())
    return n, v, bool(standard < v and n == 0)


def hard_check(probs, per_client, counts):
    n_filtered = 0
    for c in client_ids:
        n, v, filt = decide(probs[c])
        on, ov = per_client[c]
        if n != on or abs(v - ov) > 1e-3:  # reference logs tensors to 4 d.p.
            raise RuntimeError(f'client {c}: ours (out_n={n}, out_val={v:.4f}) != reference (out_n={on}, out_val={ov})')
        n_filtered += filt
    att_filtered = decide(probs[attacker_id])[2]
    if counts[1] != n_filtered or counts[2] != int(att_filtered):
        raise RuntimeError(f'filtered counts differ: ours (all={n_filtered}, attacker={int(att_filtered)}) '
                           f'vs reference (all={counts[1]}, correct={counts[2]})')
    return att_filtered


# ---------------------------------------------------------------- grid
print('training benign clients (fixed across the grid)...')
benign_w = {c: local_train(c, False, base_params, args.seed + c) for c in benign_ids}

FIELDS = ['ckpt_epoch', 'round_epoch', 'rho', 'poison_rate', 'poison_epochs', 'rep', 'filtered', 'att_argmax',
          'att_stat', 'att_p0_mean', 'att_frac_gt05', 'att_local_clean_acc', 'att_local_asr',
          'benign_p0_mean_avg', 'benign_frac_gt05_avg', 'benign_stat_max', 'n_filtered_total', 'oracle_check']
rows, probe_store, meta = [], [], []
csv_path = os.path.join(args.outdir, 'grid.csv')
t0 = time.time()
with open(csv_path, 'w', newline='') as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
    writer.writeheader()
    for ri, rate in enumerate(args.rates):
        for ei, pe in enumerate(args.poison_epochs):
            for rep in range(args.reps):
                cell_seed = args.seed + 1000 * ri + 100 * ei + rep
                cp = copy.deepcopy(base_params)
                cp['poison_rate'], cp['poison_epochs'] = rate, pe
                w_att = local_train(attacker_id, True, cp, cell_seed)
                w_by_client = dict(benign_w)
                w_by_client[attacker_id] = w_att

                g_imgs, per_client, counts = run_reference_round(w_by_client, cell_seed + 7)
                probs = score(w_by_client, g_imgs[0])
                filtered = hard_check(probs, per_client, counts)  # raises on any mismatch

                att = probs[attacker_id]
                ben = [probs[c] for c in benign_ids]
                a_n, a_v, _ = decide(att)
                net = ResNet8(num_classes=10).to(device)
                net.load_state_dict(w_att)
                acc = real_normal_test(net, handle.test_data, base_params, handle, poison=False)
                asr = real_normal_test(net, handle.test_un_target_label_data, base_params, handle, poison=True)

                row = {'ckpt_epoch': ckpt_epoch, 'round_epoch': round_epoch, 'rho': standard, 'poison_rate': rate,
                       'poison_epochs': pe, 'rep': rep, 'filtered': int(filtered), 'att_argmax': a_n,
                       'att_stat': a_v, 'att_p0_mean': float(att[:, 0].mean()),
                       'att_frac_gt05': float((att[:, 0] > 0.5).mean()), 'att_local_clean_acc': acc,
                       'att_local_asr': asr, 'benign_p0_mean_avg': float(np.mean([b[:, 0].mean() for b in ben])),
                       'benign_frac_gt05_avg': float(np.mean([(b[:, 0] > 0.5).mean() for b in ben])),
                       'benign_stat_max': max(decide(b)[1] for b in ben), 'n_filtered_total': counts[1],
                       'oracle_check': 'pass'}
                writer.writerow(row)
                fcsv.flush()
                rows.append(row)
                probe_store.append(np.stack([probs[c] for c in client_ids]).astype(np.float32))  # (10, 32, 10)
                meta.append((rate, pe, rep))
                np.savez(os.path.join(args.outdir, 'probes.npz'), probs=np.stack(probe_store),
                         poison_rate=np.array([m[0] for m in meta]), poison_epochs=np.array([m[1] for m in meta]),
                         rep=np.array([m[2] for m in meta]), client_ids=np.array(client_ids),
                         attacker_index=len(client_ids) - 1, rho=standard, ckpt_epoch=ckpt_epoch)
                print(f'[{time.time() - t0:7.0f}s] rate={rate:.2f} pe={pe:2d} rep={rep} filtered={int(filtered)} '
                      f'p0_mean={row["att_p0_mean"]:.3f} frac>0.5={row["att_frac_gt05"]:.2f} '
                      f'local_asr={asr:.1f} (oracle check passed)', flush=True)

print(f'Done: {len(rows)} cells, all passed the reference check. Wrote {csv_path}')
