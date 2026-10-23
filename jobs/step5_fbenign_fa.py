"""F-benign: FEAT_ALIGN vs s3 for benign + adversarial clients.

BN audit: check11 FEAT_ALIGN was computed in eval() mode. This script
re-runs the same measurement in train() mode (FilterFL's actual mode) and
reports both so the discrepancy is quantified.

All Stage-3 numbers here use:
  - m.train()              -- BN uses probe-batch statistics (matches algorithm.py)
  - s3 = max(softmax(m(g_imgs[0])).mean(0))  with argmax==0 check
  - GAN probe (same as FilterFL default)

Clients:
  adversarial:     34, 36, 45, 47  (rates 0.3 and 0.5)
  benign_skewed:   73, 35, 89, 49, 92, 31, 55, 59  (high class-0 share)
  benign_balanced: 1, 29, 66, 90   (class-0 share = 0.0)

Usage:
  python3 step5_fbenign_fa.py \\
      --code-dir ~/tba_attack_test/FilterFL \\
      --warmup   ~/ffl_check1/warmup_ckpt.pt \\
      --out      ~/ffl_check_fbenign
"""

import argparse, copy, os, sys, csv
import torch
import torch.nn.functional as F
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',  required=True)
ap.add_argument('--warmup',    required=True)
ap.add_argument('--out',       required=True)
ap.add_argument('--adv-clients',     type=int, nargs='+', default=[34, 36, 45, 47])
ap.add_argument('--adv-rates',       type=float, nargs='+', default=[0.3, 0.5])
ap.add_argument('--skewed-clients',  type=int, nargs='+', default=[73, 35, 89, 49, 92, 31, 55, 59])
ap.add_argument('--balanced-clients',type=int, nargs='+', default=[1, 29, 66, 90])
ap.add_argument('--local-epochs',    type=int, default=2)
args = ap.parse_args()

sys.path.insert(0, args.code_dir)
os.makedirs('saved_models', exist_ok=True)

import datetime
import yaml
import main as ref_main
import train as train_mod
from gan import gan_defense
from handle import Handle
from models.ResNet8 import ResNet8
from device import device

TARGET = 0

with open(os.path.join(args.code_dir, 'cifar_params.yaml')) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

params['number_of_total_participants'] = 100
params['num_clients']   = 10
params['poison_client_num'] = 10
params['local_epochs']  = args.local_epochs
params['lr']            = 0.01
params['momentum']      = 0.9
params['epochs']        = 1
params['seed']          = 1
params['is_poison']     = True
params['trigger_type']  = 'checker'
params['trigger_amplitude_raw'] = 0.1145

handle = Handle(
    current_time=datetime.datetime.now().strftime('%b.%d_%H.%M.%S'),
    params=params,
    name=params.get('name', format(params['type']))
)
handle.load_data()
handle.create_model()
handle.model.load_state_dict(torch.load(args.warmup, map_location=device, weights_only=False))
handle.model.to(device)
print(f'Warmup checkpoint loaded: {args.warmup}', flush=True)

global_state = copy.deepcopy(handle.model.state_dict())


def get_class0_share(client_id):
    _, loader = handle.train_data[client_id]
    total, c0 = 0, 0
    for _, labels in loader:
        c0 += (labels == TARGET).sum().item()
        total += labels.size(0)
    return c0 / max(total, 1)


def penultimate(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out); out = model.layer2(out); out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def train_client(client_id, poison=False, poison_rate=0.0):
    params_copy = dict(params)
    params_copy['poison_rate'] = poison_rate
    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    w_local = train_mod.standard_train(
        epoch=1,
        dataset_size=handle.clients_data_num[client_id],
        client=client_id,
        params=params_copy,
        model=model,
        dataset=handle.train_data[client_id],
        handle=handle,
        poison=poison,
    )
    return w_local


def measure_train_mode(global_st, local_st):
    """Compute (s3, fa) in TRAIN() mode -- matches FilterFL algorithm.py Stage 3.

    s3 = max(softmax(m_train(g_imgs[0])).mean(0))  iff argmax == TARGET, else 0.
    fa = cos(penultimate(m_train, g_imgs[0]).mean(0), W_fc[TARGET]).
    BN uses batch statistics from the 32-image probe batch.
    """
    g_imgs = gan_defense(global_st, local_st, params, the_epoch=1)
    probe = g_imgs[0]   # (32, 3, 32, 32) normalised

    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_st)
    m.train()   # BN uses probe-batch statistics -- matches algorithm.py ResNet8_model (never .eval())
    with torch.no_grad():
        out = torch.softmax(m(probe), dim=1).mean(0)
        out_val, out_n = torch.max(out, dim=0)
        s3 = out_val.item() if out_n.item() == TARGET else 0.0
        feats = penultimate(m, probe)
        fa = F.cosine_similarity(
            feats.mean(0).unsqueeze(0),
            m.linear1.weight[TARGET].unsqueeze(0)
        ).item()
    return s3, fa


def measure_eval_mode(global_st, local_st):
    """Same but in EVAL() mode -- replicates check11's (incorrect) computation for audit."""
    g_imgs = gan_defense(global_st, local_st, params, the_epoch=1)
    probe = g_imgs[0]

    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(local_st)
    m.eval()   # eval mode -- check11's original mode (wrong)
    with torch.no_grad():
        out = torch.softmax(m(probe), dim=1).mean(0)
        out_val, out_n = torch.max(out, dim=0)
        s3_eval = out_val.item() if out_n.item() == TARGET else 0.0
        feats = penultimate(m, probe)
        fa_eval = F.cosine_similarity(
            feats.mean(0).unsqueeze(0),
            m.linear1.weight[TARGET].unsqueeze(0)
        ).item()
    return s3_eval, fa_eval


rows = []
COLS = ['client', 'type', 'rate', 'class0_share',
        'fa_train', 's3_train', 'fa_eval', 's3_eval']

print(f'\n{"client":>8}  {"type":>18}  {"rate":>5}  {"share":>7}  '
      f'{"fa_train":>10}  {"s3_train":>10}  {"fa_eval":>9}  {"s3_eval":>9}')
print('-'*90)

# ── Adversarial clients ──────────────────────────────────────────────────────
for cid in args.adv_clients:
    share = get_class0_share(cid)
    for rate in args.adv_rates:
        w_local = train_client(cid, poison=True, poison_rate=rate)
        s3_tr, fa_tr = measure_train_mode(global_state, w_local)
        s3_ev, fa_ev = measure_eval_mode(global_state, w_local)
        tag = 'adversarial'
        print(f'{cid:>8}  {tag:>18}  {rate:>5.2f}  {share:>7.4f}  '
              f'{fa_tr:>10.4f}  {s3_tr:>10.4f}  {fa_ev:>9.4f}  {s3_ev:>9.4f}',
              flush=True)
        rows.append({'client': cid, 'type': tag, 'rate': rate, 'class0_share': share,
                     'fa_train': fa_tr, 's3_train': s3_tr,
                     'fa_eval': fa_ev, 's3_eval': s3_ev})

# ── Benign class-skewed clients ───────────────────────────────────────────────
for cid in args.skewed_clients:
    share = get_class0_share(cid)
    w_local = train_client(cid, poison=False)
    s3_tr, fa_tr = measure_train_mode(global_state, w_local)
    s3_ev, fa_ev = measure_eval_mode(global_state, w_local)
    tag = 'benign_skewed'
    print(f'{cid:>8}  {tag:>18}  {"—":>5}  {share:>7.4f}  '
          f'{fa_tr:>10.4f}  {s3_tr:>10.4f}  {fa_ev:>9.4f}  {s3_ev:>9.4f}',
          flush=True)
    rows.append({'client': cid, 'type': tag, 'rate': None, 'class0_share': share,
                 'fa_train': fa_tr, 's3_train': s3_tr,
                 'fa_eval': fa_ev, 's3_eval': s3_ev})

# ── Benign balanced clients ───────────────────────────────────────────────────
for cid in args.balanced_clients:
    share = get_class0_share(cid)
    w_local = train_client(cid, poison=False)
    s3_tr, fa_tr = measure_train_mode(global_state, w_local)
    s3_ev, fa_ev = measure_eval_mode(global_state, w_local)
    tag = 'benign_balanced'
    print(f'{cid:>8}  {tag:>18}  {"—":>5}  {share:>7.4f}  '
          f'{fa_tr:>10.4f}  {s3_tr:>10.4f}  {fa_ev:>9.4f}  {s3_ev:>9.4f}',
          flush=True)
    rows.append({'client': cid, 'type': tag, 'rate': None, 'class0_share': share,
                 'fa_train': fa_tr, 's3_train': s3_tr,
                 'fa_eval': fa_ev, 's3_eval': s3_ev})

# ── Save CSV ──────────────────────────────────────────────────────────────────
os.makedirs(args.out, exist_ok=True)
out_csv = os.path.join(args.out, 'fbenign_fa.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader(); w.writerows(rows)
print(f'\nCSV saved: {out_csv}', flush=True)

# ── Correlations ─────────────────────────────────────────────────────────────
def group(typ):
    return [r for r in rows if r['type'] == typ]

def spearman(subset, x_key, y_key):
    xs = [r[x_key] for r in subset if r[x_key] is not None and r[y_key] is not None]
    ys = [r[y_key] for r in subset if r[x_key] is not None and r[y_key] is not None]
    if len(xs) < 3:
        return float('nan'), float('nan')
    r, p = stats.spearmanr(xs, ys)
    return r, p

print('\n=== Spearman(fa_train, s3_train)  [TRAIN mode -- FilterFL-correct] ===')
for tag in ['adversarial', 'benign_skewed', 'benign_balanced']:
    sub = group(tag)
    r, p = spearman(sub, 'fa_train', 's3_train')
    print(f'  {tag:>18}  n={len(sub):2d}  r={r:+.3f}  p={p:.3f}')
all_ben = [r for r in rows if r['type'].startswith('benign')]
r, p = spearman(all_ben, 'fa_train', 's3_train')
print(f'  {"all_benign":>18}  n={len(all_ben):2d}  r={r:+.3f}  p={p:.3f}')
all_rows = rows
r, p = spearman(all_rows, 'fa_train', 's3_train')
print(f'  {"all_pooled":>18}  n={len(all_rows):2d}  r={r:+.3f}  p={p:.3f}')

print('\n=== BN-mode audit: EVAL vs TRAIN (check11 used eval) ===')
print(f'  {"client":>8}  {"type":>18}  {"fa_eval":>9}  {"fa_train":>9}  {"Δfa":>7}  {"s3_eval":>9}  {"s3_train":>9}')
for r_ in rows:
    dfa = r_['fa_train'] - r_['fa_eval']
    ds3 = r_['s3_train'] - r_['s3_eval']
    print(f'  {r_["client"]:>8}  {r_["type"]:>18}  '
          f'{r_["fa_eval"]:>9.4f}  {r_["fa_train"]:>9.4f}  {dfa:>+7.4f}  '
          f'{r_["s3_eval"]:>9.4f}  {r_["s3_train"]:>9.4f}')

print('\nDone. BN mode for all numbers above: TRAIN (FilterFL-correct).')
print('Eval-mode numbers shown only for check11 audit comparison.')
