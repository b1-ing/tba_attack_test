#!/usr/bin/env python3
"""Check 19: FLTrust trust score measurement across trigger types.

FLTrust (NDSS 2021) aggregates client updates weighted by:
  TS_i = ReLU(cos(g_i, g_s))
where g_s = server's model update on its 100-image root dataset,
      g_i = client i's model update (w_local - w_global), flattened.
Clients with TS_i ≈ 0 are effectively excluded from aggregation.

Predicted blind spots:
  (A) Low poison rate: g_backdoor is small, g_clean dominates → high TS
  (B) Semantic alignment: trigger gradient direction ≈ server gradient direction

This check measures TS_i for every trigger from check16/18, plus two new
semantic triggers, at two poison rates (0.50 and 0.10).

Cross-defense comparison: also measures FilterFL feat_align and s3 so we
can show which triggers evade FilterFL, which evade FLTrust, and whether
the same trigger evades both (the paper's key claim: they don't overlap).

Triggers:
  patch          — FilterFL-detected baseline
  cov_64_pos     — FilterFL-evading (sparse 64px, all +0.5)
  cov_64_randsign— FilterFL-evading (sparse 64px, random ±0.5)
  checker        — FilterFL-evading (alternating ±0.5)
  replace_global — FilterFL-evading (zero-mean cosine, additive here)
  semantic_hi    — mean(target) - mean(others), L∞=0.50 (semantic alignment)
  semantic_lo    — same, L∞=0.10 (diluted amplitude, Path A evasion)

Root dataset: 10 images per class from test set (100 total, balanced),
              server trains for same local_epoch count as clients.
4 clients × 3 seeds × 2 poison rates = 24 measurements per trigger.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',       default='FilterFL')
ap.add_argument('--warmup-dir',     default=None)
ap.add_argument('--out',            default='.')
ap.add_argument('--clients',        type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',          type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--poison-rates',   type=float, nargs='+', default=[0.50, 0.10])
ap.add_argument('--root-per-class', type=int, default=10)
ap.add_argument('--amplitude',      type=float, default=0.5)
ap.add_argument('--n-test',         type=int, default=500)
ap.add_argument('--alpha',          type=float, default=0.5)
ap.add_argument('--global-seed',    type=int, default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

import main as ref_main
import train as train_mod
from gan import gan_defense
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
N_CLASSES = 10
C, H, W = 3, 32, 32
print(f'Target class: {TARGET}', flush=True)

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'Warmup checkpoint not found: {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
global_state = copy.deepcopy(handle.model.state_dict())
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

# ── Load all test images ──────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs)   # (N, C, H, W), on CPU
all_lbls = torch.cat(all_lbls)   # (N,)

all_imgs_gpu = all_imgs.to(device)
all_lbls_gpu = all_lbls.to(device)
non_target = all_imgs_gpu[all_lbls_gpu != TARGET][:args.n_test]

# ── Root dataset: 10 images per class (100 total) ─────────────────────────────
root_imgs, root_lbls = [], []
for cls in range(N_CLASSES):
    idx = (all_lbls == cls).nonzero(as_tuple=True)[0][:args.root_per_class]
    root_imgs.append(all_imgs[idx])
    root_lbls.append(all_lbls[idx])
root_imgs = torch.cat(root_imgs).to(device)   # (100, C, H, W)
root_lbls = torch.cat(root_lbls).to(device)   # (100,)
print(f'Root dataset: {root_imgs.shape}  (target class has '
      f'{(root_lbls==TARGET).sum()} images)', flush=True)

# ── Semantic trigger: mean(target) - mean(others) ────────────────────────────
mean_target = all_imgs[all_lbls == TARGET].float().mean(0)   # (C, H, W)
mean_others = all_imgs[all_lbls != TARGET].float().mean(0)
semantic_raw = (mean_target - mean_others).numpy()
semantic_raw_amp = np.abs(semantic_raw).max()
semantic_hi = (semantic_raw / (semantic_raw_amp + 1e-9) * args.amplitude
               ).astype(np.float32)
semantic_lo = (semantic_hi * 0.2).astype(np.float32)    # L∞ = 0.10

print(f'Semantic trigger: L∞(hi)={np.abs(semantic_hi).max():.3f}  '
      f'mean={semantic_hi.mean():.4f}  std={semantic_hi.std():.4f}')

# ── Trigger patterns ─────────────────────────────────────────────────────────
_cpu = torch.device('cpu')
_patch_mask_t  = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)
_patch_mask_np = _patch_mask_t.numpy().astype(bool)
_patch_img = np.zeros((C, H, W), dtype=np.float32)
_patch_img[_patch_mask_np] = 1.0

rng64 = np.random.RandomState(42)
pos64 = rng64.choice(H * W, size=64, replace=False)
_rng_sign = np.random.RandomState(7)
_sign_map = (_rng_sign.randint(0, 2, (H, W)) * 2 - 1).astype(np.float32)

def _sparse(positions, sign_val):
    pat = np.zeros((C, H, W), dtype=np.float32)
    for p in positions:
        h_p, w_p = divmod(int(p), W)
        pat[:, h_p, w_p] = sign_val * args.amplitude
    return pat

def _sparse_rnd(positions, sign_map):
    pat = np.zeros((C, H, W), dtype=np.float32)
    for p in positions:
        h_p, w_p = divmod(int(p), W)
        for c in range(C):
            pat[c, h_p, w_p] = float(sign_map[h_p, w_p]) * args.amplitude
    return pat

rows_c = np.arange(H).reshape(H, 1); cols_c = np.arange(W).reshape(1, W)
checker_hw = ((rows_c + cols_c) % 2).astype(np.float32) * 2.0 - 1.0
checker_pat = np.stack([checker_hw * args.amplitude] * C, axis=0)

k_x = 10; col_idx = np.arange(W, dtype=np.float32)
stripe = 0.5 * np.cos(2 * np.pi * k_x * col_idx / W)
global_pat = np.stack([np.tile(stripe, (H, 1))] * C, axis=0).astype(np.float32)

TRIGGERS = [
    ('patch',           _patch_img,                          'replacement'),
    ('cov_64_pos',      _sparse(pos64, +1.0),               'additive'),
    ('cov_64_randsign', _sparse_rnd(pos64, _sign_map),      'additive'),
    ('checker',         checker_pat,                          'additive'),
    ('replace_global',  global_pat,                           'additive'),
    ('semantic_hi',     semantic_hi,                          'additive'),
    ('semantic_lo',     semantic_lo,                          'additive'),
]

# ── Utility: flatten all parameters to a single vector ───────────────────────
def flatten_delta(w_a, w_b):
    """Flatten (w_a - w_b) across all parameters into a 1D CPU tensor."""
    parts = []
    for k in w_a:
        parts.append((w_a[k].float().cpu() - w_b[k].float().cpu()).flatten())
    return torch.cat(parts)

# ── Server update on root dataset ─────────────────────────────────────────────
def compute_server_update():
    """Train global model on root dataset for local_epoch steps → return Δw_s."""
    m = ResNet8(num_classes=N_CLASSES).to(device)
    m.load_state_dict(copy.deepcopy(global_state))
    m.train()
    lr = params.get('lr', 0.01)
    momentum = params.get('momentum', 0.9)
    wd = params.get('weight_decay', 1e-4)
    local_epochs = params.get('local_epoch', 6)
    opt = optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    ds = TensorDataset(root_imgs, root_lbls)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for _ in range(local_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(m(xb), yb).backward()
            opt.step()
    w_s = {k: v.cpu() for k, v in m.state_dict().items()}
    return w_s

print('\nComputing server update on root dataset...', flush=True)
w_server = compute_server_update()
g_s = flatten_delta(w_server, {k: v.cpu() for k, v in global_state.items()})
print(f'Server update norm: {g_s.norm().item():.4f}', flush=True)

# ── Feature extractor (for FilterFL feat_align) ───────────────────────────────
def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out); out = model.layer2(out)
    out = model.layer3(out); out = model.avgpool(out)
    return torch.flatten(out, 1)

# ── Main measurement loop ─────────────────────────────────────────────────────
COLS = ['trigger', 'poison_rate', 'client', 'seed',
        'trust_score', 'feat_align', 's3', 'asr_proxy',
        'g_i_norm', 'g_i_norm_ratio']
rows = []

print(f'\n{"trigger":>16}  {"pr":>5}  {"cli":>3}  {"sd":>2}  '
      f'{"TS":>7}  {"fa":>7}  {"s3":>7}  {"asr":>7}  {"gi_norm":>8}')
print('-' * 77)

for poison_rate in args.poison_rates:
    for tname, pat_np, pat_type in TRIGGERS:
        _pat_t = torch.from_numpy(pat_np)
        for client_id in args.clients:
            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed)
                p = copy.deepcopy(params)
                p['poison_rate'] = poison_rate

                import train as tm
                _orig = tm.apply_pattern_batch_gpu

                if pat_type == 'replacement' and tname == 'patch':
                    def patched(images, pms, index, handle_, poison_mask):
                        return _orig(images, pms, index, handle_, poison_mask)
                else:
                    def patched(images, pms, index, handle_, poison_mask,
                                _pt=_pat_t):
                        pm_exp = poison_mask.view(-1, 1, 1, 1)
                        pat = _pt.to(images.device)
                        poisoned = torch.clamp(images + pat.unsqueeze(0),
                                               -1.0, 1.0)
                        labels = torch.full((images.size(0),), TARGET,
                                            device=images.device,
                                            dtype=torch.long)
                        return (torch.where(pm_exp.expand_as(images).bool(),
                                            poisoned, images),
                                labels)

                tm.apply_pattern_batch_gpu = patched
                m = ResNet8(num_classes=N_CLASSES).to(device)
                m.load_state_dict(copy.deepcopy(global_state))
                try:
                    w_local = train_mod.standard_train(
                        epoch=1,
                        dataset_size=handle.clients_data_num[client_id],
                        client=client_id,
                        params=p,
                        model=m,
                        dataset=handle.train_data[client_id],
                        handle=handle,
                        poison=True,
                    )
                finally:
                    tm.apply_pattern_batch_gpu = _orig

                # ── FLTrust trust score ───────────────────────────────────────
                g_i = flatten_delta(w_local,
                                    {k: v.cpu() for k, v in global_state.items()})
                g_i_norm = g_i.norm().item()
                g_s_norm = g_s.norm().item()
                cos_raw = F.cosine_similarity(g_i.unsqueeze(0),
                                              g_s.unsqueeze(0)).item()
                trust_score = max(0.0, cos_raw)

                # ── FilterFL metrics ──────────────────────────────────────────
                W_l_target = w_local['linear1.weight'][TARGET]
                ml = ResNet8(num_classes=N_CLASSES).to(device)
                ml.load_state_dict(w_local); ml.eval()
                torch.manual_seed(42); np.random.seed(42)
                probe_imgs = gan_defense(global_state, w_local, params,
                                         the_epoch=1)[0]
                with torch.no_grad():
                    probs = torch.softmax(ml(probe_imgs), 1)
                    s3 = probs.mean(0)[TARGET].item()
                    feats = features(ml, probe_imgs)
                fa = F.cosine_similarity(feats.mean(0, keepdim=True),
                                          W_l_target.unsqueeze(0)).item()

                # ── ASR proxy ─────────────────────────────────────────────────
                with torch.no_grad():
                    if tname == 'patch':
                        mask_b = (_patch_mask_t.to(device).bool()
                                  .unsqueeze(0).expand_as(non_target))
                        trig_imgs = torch.where(mask_b,
                                                torch.tensor(1.0, device=device),
                                                non_target)
                    else:
                        trig_imgs = torch.clamp(
                            non_target + _pat_t.to(device).unsqueeze(0),
                            -1.0, 1.0)
                    asr_proxy = torch.softmax(ml(trig_imgs), 1).mean(0)[TARGET].item()

                row = {'trigger': tname, 'poison_rate': poison_rate,
                       'client': client_id, 'seed': seed,
                       'trust_score': trust_score, 'feat_align': fa, 's3': s3,
                       'asr_proxy': asr_proxy,
                       'g_i_norm': g_i_norm,
                       'g_i_norm_ratio': g_i_norm / (g_s_norm + 1e-9)}
                rows.append(row)
                print(f'{tname:>16}  {poison_rate:>5.2f}  {client_id:>3}  '
                      f'{seed:>2}  {trust_score:>7.4f}  {fa:>7.4f}  '
                      f'{s3:>7.4f}  {asr_proxy:>7.4f}  {g_i_norm:>8.3f}',
                      flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst)/len(lst) if lst else float('nan')

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>16}  {'pr':>5}  {'TS':>7}  {'fa':>7}  "
      f"{'s3':>7}  {'asr':>7}  {'gi_ratio':>9}")
print('-' * 65)
for pr in args.poison_rates:
    for tname, _, _ in TRIGGERS:
        tr = [r for r in rows if r['trigger']==tname and r['poison_rate']==pr]
        print(f"{tname:>16}  {pr:>5.2f}  "
              f"{mn([r['trust_score'] for r in tr]):>7.4f}  "
              f"{mn([r['feat_align']  for r in tr]):>7.4f}  "
              f"{mn([r['s3']          for r in tr]):>7.4f}  "
              f"{mn([r['asr_proxy']   for r in tr]):>7.4f}  "
              f"{mn([r['g_i_norm_ratio'] for r in tr]):>9.3f}")

print('\n=== Cross-defense comparison (evasion = filter evades AND trust high) ===')
print(f"  Criteria: s3 < 0.10 (evades FilterFL), trust_score > 0.50 (survives FLTrust)")
print(f"{'trigger':>16}  {'pr':>5}  {'evades_FL':>10}  {'survives_FT':>12}  "
      f"{'both':>6}  {'asr':>7}")
for pr in args.poison_rates:
    for tname, _, _ in TRIGGERS:
        tr = [r for r in rows if r['trigger']==tname and r['poison_rate']==pr]
        s3_m  = mn([r['s3']          for r in tr])
        ts_m  = mn([r['trust_score'] for r in tr])
        asr_m = mn([r['asr_proxy']   for r in tr])
        efl = 'YES' if s3_m < 0.10 else 'no'
        sft = 'YES' if ts_m > 0.50 else 'no'
        both = 'YES!' if s3_m < 0.10 and ts_m > 0.50 else '-'
        print(f"{tname:>16}  {pr:>5.2f}  {efl:>10}  {sft:>12}  "
              f"{both:>6}  {asr_m:>7.4f}")

print('\n=== Correlations ===')
import statistics
all_ts = [r['trust_score'] for r in rows]
for col in ['feat_align', 's3', 'asr_proxy', 'g_i_norm_ratio']:
    vals = [r[col] for r in rows]
    n = len(vals); mx = statistics.mean(vals); my = statistics.mean(all_ts)
    num = sum((x-mx)*(y-my) for x, y in zip(vals, all_ts))
    denom = (sum((x-mx)**2 for x in vals) * sum((y-my)**2 for y in all_ts))**0.5
    print(f'  cor(trust_score, {col}) = {num/denom if denom>1e-12 else 0.:+.4f}')

csv_path = os.path.join(args.out, 'check19_fltrust_trust.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(rows)
print(f'\nSaved to {csv_path}')
