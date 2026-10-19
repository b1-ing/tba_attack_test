#!/usr/bin/env python3
"""Check 16: Coverage threshold sweep.

The check14 and check15 results show that spatial coverage (not frequency,
amplitude, or modification type) drives evasion of Stage-3 detection. Two
data points exist: 16px (detected, s3≈0.19) and 1024px (evades, s3≈0.03).

This sweep fills the curve: 7 coverage levels × 4 clients × 3 seeds = 84
measurements, plus the evasion_score (cos(trigger_feat, GAN_centroid) through
the global model) for each coverage level.

Triggers: additive, random pixel positions, amplitude 0.5. One fixed spatial
mask per coverage level (mask_seed = coverage_level index) so the evasion_score
is a clean scalar per level.

Reference triggers included for anchoring:
  - patch:      16px, replacement, fixed positions (known: s3≈0.19, evasion_score≈0.990)
  - checker:    1024px, additive ±0.5 (known: s3≈0.00, evasion_score≈low)
  - cov_16 …   Random 16px additive (tests whether POSITIONS matter vs patch)
  - cov_1024    Random 1024px additive (sanity-check against checker)

Expected outcome:
  - evasion_score decreases monotonically with coverage
  - s3 tracks evasion_score: drops below detection threshold at some coverage N*
  - N* is the paper's main quantitative result
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',    default='FilterFL')
ap.add_argument('--warmup-dir',  default=None)
ap.add_argument('--out',         default='.')
ap.add_argument('--clients',     type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',       type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--amplitude',   type=float, default=0.5)
ap.add_argument('--n-test',      type=int, default=500)
ap.add_argument('--alpha',       type=float, default=0.5)
ap.add_argument('--global-seed', type=int, default=1)
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
print(f'Target class: {TARGET}', flush=True)

warmup_dir = args.warmup_dir or os.path.expanduser('~/ffl_check1')
ckpt = os.path.join(warmup_dir, 'warmup_ckpt.pt')
assert os.path.exists(ckpt), f'Warmup checkpoint not found: {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
global_state = copy.deepcopy(handle.model.state_dict())
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

# Test pool for ASR proxy
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs).to(device)
all_lbls = torch.cat(all_lbls).to(device)
non_target = all_imgs[all_lbls != TARGET][:args.n_test]
print(f'ASR eval pool: {non_target.shape}', flush=True)

C, H, W = 3, 32, 32

# ── Trigger construction ──────────────────────────────────────────────────────

_cpu = torch.device('cpu')
_patch_mask_t  = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)
_patch_mask_np = _patch_mask_t.numpy().astype(bool)

COVERAGE_LEVELS = [16, 32, 64, 128, 256, 512, 1024]

def make_coverage_pattern(n_pixels, mask_seed):
    """Additive trigger at n_pixels random SPATIAL positions (same across channels)."""
    rng = np.random.RandomState(mask_seed)
    spatial = rng.choice(H * W, size=n_pixels, replace=False)
    pat = np.zeros((C, H, W), dtype=np.float32)
    for pos in spatial:
        h_pos, w_pos = divmod(int(pos), W)
        pat[:, h_pos, w_pos] = args.amplitude
    return pat

# Pre-build all coverage patterns (one mask per coverage level, fixed mask_seed)
coverage_patterns = {}
for idx, n in enumerate(COVERAGE_LEVELS):
    coverage_patterns[n] = make_coverage_pattern(n, mask_seed=idx)

# Checker reference
rows_c = np.arange(H).reshape(H, 1)
cols_c = np.arange(W).reshape(1, W)
checker_hw = ((rows_c + cols_c) % 2).astype(np.float32) * 2.0 - 1.0
checker_pat = np.stack([checker_hw * args.amplitude] * C, axis=0)

# ── Feature extractor ─────────────────────────────────────────────────────────
def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)

# ── Evasion scores through global model (once per trigger) ────────────────────
print('\n=== Evasion scores through global model ===')
print(f"{'trigger':>20}  {'n_px':>6}  {'evasion_score':>14}  "
      f"(low = outside GAN manifold = evades)")
print('-' * 55)

g_model = ResNet8(num_classes=10).to(device)
g_model.load_state_dict(global_state)
g_model.eval()

torch.manual_seed(42); np.random.seed(42)
probe_imgs_g = gan_defense(global_state, global_state, params, the_epoch=1)[0]
with torch.no_grad():
    probe_centroid_g = features(g_model, probe_imgs_g).mean(0)  # (512,)

evasion_scores = {}

def compute_evasion_score(pat_np, label, n_px):
    pat_t = torch.from_numpy(pat_np).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = features(g_model, pat_t).squeeze(0)
    score = F.cosine_similarity(feat.unsqueeze(0),
                                probe_centroid_g.unsqueeze(0)).item()
    evasion_scores[label] = score
    print(f'{label:>20}  {n_px:>6}  {score:>14.4f}')
    return score

# patch reference
_patch_img = np.zeros((C, H, W), dtype=np.float32)
_patch_img[_patch_mask_np] = 1.0
compute_evasion_score(_patch_img, 'patch', 16)
compute_evasion_score(checker_pat, 'checker', 1024)

for n in COVERAGE_LEVELS:
    compute_evasion_score(coverage_patterns[n], f'cov_{n}', n)

# ── Measurement function ───────────────────────────────────────────────────────
TRIGGERS = ['patch'] + [f'cov_{n}' for n in COVERAGE_LEVELS] + ['checker']
COLS = ['trigger', 'n_pixels', 'client', 'seed',
        'evasion_score', 'feat_align', 's3', 'asr_proxy']

rows = []
print(f'\n{"trigger":>12}  {"n_px":>6}  {"client":>6}  {"seed":>4}  '
      f'{"ev_sc":>7}  {"feat_al":>8}  {"s3":>7}  {"asr_px":>8}')
print('-' * 72)


def measure(trigger_name, n_pixels, pat_np, client_id, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    p = copy.deepcopy(params)
    p['poison_rate'] = 0.50

    import train as tm
    _orig = tm.apply_pattern_batch_gpu

    if trigger_name == 'patch':
        def patched(images, pms, index, handle_, poison_mask):
            return _orig(images, pms, index, handle_, poison_mask)
    else:
        _pat_t = torch.from_numpy(pat_np)
        _is_replacement = False  # all coverage triggers are additive

        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat = _pat_t.to(images.device)
            poisoned = torch.clamp(images + pat.unsqueeze(0), -1.0, 1.0)
            labels = torch.full((images.size(0),), TARGET,
                                device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), labels

    tm.apply_pattern_batch_gpu = patched
    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))
    try:
        w_local = train_mod.standard_train(
            epoch=1,
            dataset_size=handle.clients_data_num[client_id],
            client=client_id,
            params=p,
            model=model,
            dataset=handle.train_data[client_id],
            handle=handle,
            poison=True,
        )
    finally:
        tm.apply_pattern_batch_gpu = _orig

    W_l_target = w_local['linear1.weight'][TARGET]
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(w_local)
    m.eval()

    torch.manual_seed(42); np.random.seed(42)
    probe_imgs = gan_defense(global_state, w_local, params, the_epoch=1)[0]
    with torch.no_grad():
        probs = torch.softmax(m(probe_imgs), 1)
        s3 = probs.mean(0)[TARGET].item()
        feats = features(m, probe_imgs)
    feat_align = F.cosine_similarity(
        feats.mean(0, keepdim=True), W_l_target.unsqueeze(0)).item()

    # ASR proxy
    with torch.no_grad():
        if trigger_name == 'patch':
            mask_b = _patch_mask_t.to(device).bool().unsqueeze(0).expand_as(non_target)
            triggered = torch.where(mask_b,
                                    torch.tensor(1.0, device=device), non_target)
        else:
            pat = torch.from_numpy(pat_np).to(device)
            triggered = torch.clamp(non_target + pat.unsqueeze(0), -1.0, 1.0)
        asr_proxy = torch.softmax(m(triggered), 1).mean(0)[TARGET].item()

    ev_sc = evasion_scores.get(trigger_name, float('nan'))
    print(f'{trigger_name:>12}  {n_pixels:>6}  {client_id:>6}  {seed:>4}  '
          f'{ev_sc:>7.4f}  {feat_align:>8.4f}  {s3:>7.4f}  {asr_proxy:>8.4f}',
          flush=True)

    return {'trigger': trigger_name, 'n_pixels': n_pixels,
            'client': client_id, 'seed': seed,
            'evasion_score': ev_sc, 'feat_align': feat_align,
            's3': s3, 'asr_proxy': asr_proxy}


trigger_list = [
    ('patch',   16,   _patch_img),
] + [
    (f'cov_{n}', n, coverage_patterns[n]) for n in COVERAGE_LEVELS
] + [
    ('checker', 1024, checker_pat),
]

for tname, n_px, pat in trigger_list:
    for client_id in args.clients:
        for seed in args.seeds:
            rows.append(measure(tname, n_px, pat, client_id, seed))

# ── Summary ───────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst)/len(lst) if lst else float('nan')

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>12}  {'n_px':>6}  {'ev_score':>9}  "
      f"{'feat_al':>8}  {'s3':>7}  {'asr_px':>8}")
print('-' * 60)
for tname, n_px, _ in trigger_list:
    tr = [r for r in rows if r['trigger'] == tname]
    print(f"{tname:>12}  {n_px:>6}  "
          f"{mn([r['evasion_score'] for r in tr]):>9.4f}  "
          f"{mn([r['feat_align']    for r in tr]):>8.4f}  "
          f"{mn([r['s3']            for r in tr]):>7.4f}  "
          f"{mn([r['asr_proxy']     for r in tr]):>8.4f}")

# Find transition point
print('\n=== Coverage threshold ===')
for tname, n_px, _ in trigger_list:
    tr = [r for r in rows if r['trigger'] == tname]
    s3_mean = mn([r['s3'] for r in tr])
    ev_mean = mn([r['evasion_score'] for r in tr])
    if s3_mean < 0.10:
        print(f'  Evasion at {n_px}px: s3={s3_mean:.4f}, evasion_score={ev_mean:.4f}')
    else:
        print(f'  Detected at {n_px}px: s3={s3_mean:.4f}, evasion_score={ev_mean:.4f}')

csv_path = os.path.join(args.out, 'check16_coverage_sweep.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(rows)
print(f'\nSaved to {csv_path}')
