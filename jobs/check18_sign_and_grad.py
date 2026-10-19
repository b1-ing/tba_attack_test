#!/usr/bin/env python3
"""Check 18: Sign pattern and gradient-guided position selection at k=64px.

Check 17 showed that weight-space rank metrics don't predict feat_align.
The cov_1024 (all +0.5, detected) vs checker (±0.5, evades) result implies
sign pattern matters. Check 16 only ever tested all-positive triggers.

This experiment isolates two degrees of freedom at fixed k=64px:

  1. Sign pattern:
       cov_64_pos      — 64px, all +0.5  (check16 baseline: feat_align≈0.048)
       cov_64_neg      — 64px, all -0.5  (symmetry check)
       cov_64_randsign — 64px, random ±0.5 per position (zero-mean)

  2. Gradient-guided position selection (using global model sensitivity):
       cov_64_grad_lo  — 64px, positions with LOWEST ∂P(target)/∂pixel
                         (least sensitive → model can't easily respond to these)
       cov_64_grad_hi  — 64px, positions with HIGHEST ∂P(target)/∂pixel
                         (most sensitive → model responds strongly to these)
       cov_64_grad_sign — grad_lo positions, sign set by gradient direction
                         (combined: position + sign optimised together)

  Also includes Version A feasibility test:
       opt_20steps     — start from random 64px trigger, take 20 gradient steps
                         on T to minimise feat_align (fixed local model, no re-train),
                         then re-train and measure if feat_align is lower.

All at k=64, amplitude 0.5, 4 clients × 3 seeds = 12 measurements per trigger.
6 triggers × 12 = 72 measurements total.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',    default='FilterFL')
ap.add_argument('--warmup-dir',  default=None)
ap.add_argument('--out',         default='.')
ap.add_argument('--clients',     type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',       type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--k',           type=int, default=64)
ap.add_argument('--amplitude',   type=float, default=0.5)
ap.add_argument('--opt-steps',   type=int, default=20)
ap.add_argument('--opt-lr',      type=float, default=0.01)
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

all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs).to(device)
all_lbls = torch.cat(all_lbls).to(device)
non_target = all_imgs[all_lbls != TARGET][:args.n_test]

C, H, W = 3, 32, 32
K = args.k
AMP = args.amplitude

# ── Feature extractor ─────────────────────────────────────────────────────────
def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out); out = model.layer2(out)
    out = model.layer3(out); out = model.avgpool(out)
    return torch.flatten(out, 1)

# ── Gradient sensitivity map of global model ──────────────────────────────────
g_model = ResNet8(num_classes=10).to(device)
g_model.load_state_dict(global_state)
g_model.eval()

# Use non-target images as reference; average gradient magnitude over 50 samples
ref = non_target[:50].clone().requires_grad_(True)
logits = g_model(ref)
target_prob = torch.softmax(logits, 1)[:, TARGET].sum()
target_prob.backward()
# Per-pixel sensitivity: mean abs gradient across channels and reference images
sens_map = ref.grad.detach().abs().mean(dim=(0, 1))   # (H, W)
sens_flat = sens_map.flatten()                          # (H*W,)
print(f'Sensitivity map: min={sens_flat.min():.4f}  max={sens_flat.max():.4f}', flush=True)

# Gradient direction at high-sensitivity positions (for grad_sign trigger)
grad_dir = ref.grad.detach().mean(dim=0)   # (C, H, W) — mean signed gradient

# ── Build trigger patterns ────────────────────────────────────────────────────
# Fixed spatial positions (same across all seeds for reproducibility)
rng_pos = np.random.RandomState(42)
rand_positions = rng_pos.choice(H * W, size=K, replace=False)   # (K,)

# Gradient-selected positions
grad_lo_pos = sens_flat.argsort()[:K].cpu().numpy()    # K least sensitive
grad_hi_pos = sens_flat.argsort()[-K:].cpu().numpy()   # K most sensitive

def make_pattern(positions, sign_fn):
    """Build (C, H, W) pattern from flat spatial positions and per-position sign fn."""
    pat = np.zeros((C, H, W), dtype=np.float32)
    for pos in positions:
        h_pos, w_pos = divmod(int(pos), W)
        for c in range(C):
            pat[c, h_pos, w_pos] = sign_fn(c, h_pos, w_pos) * AMP
    return pat

def const_pos(c, h, w):  return +1.0
def const_neg(c, h, w):  return -1.0

_rng_sign = np.random.RandomState(7)
_sign_map = (_rng_sign.randint(0, 2, (H, W)) * 2 - 1).astype(np.float32)
def rand_sign(c, h, w):  return float(_sign_map[h, w])

def grad_sign(c, h, w):
    # Sign that pushes activation in direction that REDUCES P(target)
    # = negative of gradient direction (we want to avoid what the model finds salient)
    return -float(np.sign(grad_dir[c, h, w].cpu().item() + 1e-9))

patterns = {
    'cov_64_pos':       make_pattern(rand_positions, const_pos),
    'cov_64_neg':       make_pattern(rand_positions, const_neg),
    'cov_64_randsign':  make_pattern(rand_positions, rand_sign),
    'cov_64_grad_lo':   make_pattern(grad_lo_pos,   const_pos),
    'cov_64_grad_hi':   make_pattern(grad_hi_pos,   const_pos),
    'cov_64_grad_sign': make_pattern(grad_lo_pos,   grad_sign),
}

# Print spectral medians for reference
def median_f(pat):
    k_max = np.sqrt((H/2)**2 + (W/2)**2)
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    KX, KY = np.meshgrid(kx, ky)
    K_norm = np.sqrt(KX**2 + KY**2) / k_max
    power = sum(np.abs(np.fft.fftshift(np.fft.fft2(pat[ch])))**2 for ch in range(C)) / C
    edges = np.linspace(0, 1, 101); centers = 0.5*(edges[:-1]+edges[1:])
    bin_p = np.array([(power[(K_norm>=lo)&(K_norm<hi)].sum()
                       if ((K_norm>=lo)&(K_norm<hi)).any() else 0.)
                      for lo, hi in zip(edges[:-1], edges[1:])])
    cum = np.cumsum(bin_p)
    return float(centers[min(np.searchsorted(cum/(cum[-1]+1e-30), 0.5), 99)])

print('\n=== Pattern properties ===')
print(f"{'trigger':>20}  {'mean':>7}  {'std':>7}  {'median_f':>9}  {'L∞':>5}")
for name, pat in patterns.items():
    print(f'{name:>20}  {pat.mean():>7.4f}  {pat.std():>7.4f}  '
          f'{median_f(pat):>9.4f}  {np.abs(pat).max():>5.3f}')

# ── Version A: gradient optimisation of trigger (feasibility test) ────────────
print('\n=== Version A: gradient optimisation feasibility ===')
print(f'Optimising trigger for {args.opt_steps} steps on fixed local model...', flush=True)

# Train a reference local model with the random-pos all-positive trigger
def train_one(pat_np, client_id, seed, trigger_name='_tmp'):
    torch.manual_seed(seed); np.random.seed(seed)
    p = copy.deepcopy(params); p['poison_rate'] = 0.50
    import train as tm
    _orig = tm.apply_pattern_batch_gpu
    _pat_t = torch.from_numpy(pat_np)
    def patched(images, pms, index, handle_, poison_mask):
        pm_exp = poison_mask.view(-1, 1, 1, 1)
        pat = _pat_t.to(images.device)
        poisoned = torch.clamp(images + pat.unsqueeze(0), -1.0, 1.0)
        labels = torch.full((images.size(0),), TARGET,
                            device=images.device, dtype=torch.long)
        return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), labels
    tm.apply_pattern_batch_gpu = patched
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(copy.deepcopy(global_state))
    try:
        w = train_mod.standard_train(
            epoch=1, dataset_size=handle.clients_data_num[client_id],
            client=client_id, params=p, model=m,
            dataset=handle.train_data[client_id], handle=handle, poison=True)
    finally:
        tm.apply_pattern_batch_gpu = _orig
    return w

def compute_feat_align(w_local):
    m = ResNet8(num_classes=10).to(device); m.load_state_dict(w_local); m.eval()
    W_l_target = w_local['linear1.weight'].to(device)[TARGET]
    torch.manual_seed(42); np.random.seed(42)
    probe_imgs = gan_defense(global_state, w_local, params, the_epoch=1)[0]
    with torch.no_grad():
        probs = torch.softmax(m(probe_imgs), 1)
        s3 = probs.mean(0)[TARGET].item()
        feats = features(m, probe_imgs)
    fa = F.cosine_similarity(feats.mean(0, keepdim=True),
                              W_l_target.unsqueeze(0)).item()
    return fa, s3

# Reference: train with pos trigger on client 34 seed 1
ref_client, ref_seed = 34, 1
w_ref = train_one(patterns['cov_64_pos'], ref_client, ref_seed)
fa_before, s3_before = compute_feat_align(w_ref)
print(f'Before opt: feat_align={fa_before:.4f}  s3={s3_before:.4f}')

# Now do gradient optimisation on T for a fixed model
m_fixed = ResNet8(num_classes=10).to(device)
m_fixed.load_state_dict(w_ref)
m_fixed.eval()

T_opt = torch.tensor(patterns['cov_64_pos'], device=device, requires_grad=True)
opt = torch.optim.Adam([T_opt], lr=args.opt_lr)

torch.manual_seed(42); np.random.seed(42)
probe_for_opt = gan_defense(global_state, w_ref, params, the_epoch=1)[0]
W_target_fixed = w_ref['linear1.weight'].to(device)[TARGET]

fa_traj = []
for step in range(args.opt_steps):
    opt.zero_grad()
    # Apply T_opt to probe images (simulate what triggered training induces)
    T_clamped = T_opt.clamp(-AMP, AMP)
    perturbed = torch.clamp(probe_for_opt + T_clamped.unsqueeze(0), -1.0, 1.0)
    feats_p = features(m_fixed, perturbed)
    fa_loss = F.cosine_similarity(feats_p.mean(0, keepdim=True),
                                   W_target_fixed.unsqueeze(0))
    fa_loss.backward()
    opt.step()
    fa_traj.append(fa_loss.item())
    if step % 5 == 0:
        print(f'  step {step:3d}: proxy_feat_align={fa_loss.item():.4f}', flush=True)

# Extract optimised trigger, L∞ normalise
T_opt_np = T_opt.detach().cpu().numpy()
T_opt_np = T_opt_np / (np.abs(T_opt_np).max() + 1e-9) * AMP
patterns['opt_20steps'] = T_opt_np.astype(np.float32)
print(f'Optimised trigger: mean={T_opt_np.mean():.4f}  std={T_opt_np.std():.4f}')

# Retrain with optimised trigger and measure
w_opt = train_one(T_opt_np, ref_client, ref_seed)
fa_after, s3_after = compute_feat_align(w_opt)
print(f'After opt (retrain): feat_align={fa_after:.4f}  s3={s3_after:.4f}')
print(f'Improvement: Δfeat_align={fa_after-fa_before:+.4f}  Δs3={s3_after-s3_before:+.4f}')
print('Gradient opt is USEFUL' if fa_after < fa_before - 0.01
      else 'Gradient opt gives NO benefit over random')

# ── Main measurement loop ─────────────────────────────────────────────────────
TRIGGER_NAMES = list(patterns.keys())
COLS = ['trigger', 'client', 'seed', 'feat_align', 's3', 'asr_proxy']
rows = []

print(f'\n{"trigger":>22}  {"client":>6}  {"seed":>4}  '
      f'{"feat_al":>8}  {"s3":>7}  {"asr_px":>8}')
print('-' * 65)

for tname in TRIGGER_NAMES:
    pat_np = patterns[tname]
    _pat_t = torch.from_numpy(pat_np)
    for client_id in args.clients:
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            p = copy.deepcopy(params); p['poison_rate'] = 0.50
            import train as tm
            _orig = tm.apply_pattern_batch_gpu
            def patched(images, pms, index, handle_, poison_mask,
                        _pt=_pat_t):
                pm_exp = poison_mask.view(-1, 1, 1, 1)
                pat = _pt.to(images.device)
                poisoned = torch.clamp(images + pat.unsqueeze(0), -1.0, 1.0)
                labels = torch.full((images.size(0),), TARGET,
                                    device=images.device, dtype=torch.long)
                return torch.where(pm_exp.expand_as(images).bool(),
                                   poisoned, images), labels
            tm.apply_pattern_batch_gpu = patched
            m = ResNet8(num_classes=10).to(device)
            m.load_state_dict(copy.deepcopy(global_state))
            try:
                w_local = train_mod.standard_train(
                    epoch=1, dataset_size=handle.clients_data_num[client_id],
                    client=client_id, params=p, model=m,
                    dataset=handle.train_data[client_id], handle=handle, poison=True)
            finally:
                tm.apply_pattern_batch_gpu = _orig

            W_l_target = w_local['linear1.weight'].to(device)[TARGET]
            ml = ResNet8(num_classes=10).to(device)
            ml.load_state_dict(w_local); ml.eval()
            torch.manual_seed(42); np.random.seed(42)
            probe_imgs = gan_defense(global_state, w_local, params, the_epoch=1)[0]
            with torch.no_grad():
                probs = torch.softmax(ml(probe_imgs), 1)
                s3 = probs.mean(0)[TARGET].item()
                feats = features(ml, probe_imgs)
            fa = F.cosine_similarity(feats.mean(0, keepdim=True),
                                      W_l_target.unsqueeze(0)).item()
            with torch.no_grad():
                trig_imgs = torch.clamp(
                    non_target + _pat_t.to(device).unsqueeze(0), -1., 1.)
                asr_proxy = torch.softmax(ml(trig_imgs), 1).mean(0)[TARGET].item()

            row = {'trigger': tname, 'client': client_id, 'seed': seed,
                   'feat_align': fa, 's3': s3, 'asr_proxy': asr_proxy}
            rows.append(row)
            print(f'{tname:>22}  {client_id:>6}  {seed:>4}  '
                  f'{fa:>8.4f}  {s3:>7.4f}  {asr_proxy:>8.4f}', flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst)/len(lst) if lst else float('nan')

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>22}  {'feat_al':>8}  {'s3':>7}  {'asr_px':>8}")
print('-' * 52)
for tname in TRIGGER_NAMES:
    tr = [r for r in rows if r['trigger'] == tname]
    print(f"{tname:>22}  {mn([r['feat_align'] for r in tr]):>8.4f}  "
          f"{mn([r['s3'] for r in tr]):>7.4f}  "
          f"{mn([r['asr_proxy'] for r in tr]):>8.4f}")

print('\n=== Key comparisons ===')
def show(a, b, label):
    fa_a = mn([r['feat_align'] for r in rows if r['trigger']==a])
    fa_b = mn([r['feat_align'] for r in rows if r['trigger']==b])
    s3_a = mn([r['s3'] for r in rows if r['trigger']==a])
    s3_b = mn([r['s3'] for r in rows if r['trigger']==b])
    print(f'  {label}: {a} feat_al={fa_a:.4f} s3={s3_a:.4f}  vs  '
          f'{b} feat_al={fa_b:.4f} s3={s3_b:.4f}')

show('cov_64_pos',      'cov_64_randsign', 'Sign pattern')
show('cov_64_pos',      'cov_64_neg',      'Sign polarity')
show('cov_64_pos',      'cov_64_grad_lo',  'Position (lo-sens)')
show('cov_64_pos',      'cov_64_grad_hi',  'Position (hi-sens)')
show('cov_64_grad_lo',  'cov_64_grad_sign','Grad position + sign')
show('cov_64_randsign', 'opt_20steps',     'Random sign vs opt')

csv_path = os.path.join(args.out, 'check18_sign_and_grad.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(rows)
print(f'\nSaved to {csv_path}')
