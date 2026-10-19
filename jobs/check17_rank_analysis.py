#!/usr/bin/env python3
"""Check 17: Effective rank analysis of the backdoor weight update.

Check 16 revealed a critical anomaly: cov_1024 (all pixels set to +0.5, uniform)
is DETECTED (s3=0.241), while checker at the same 1024px coverage EVADES
(s3=0.007). Coverage alone does not predict evasion — the sign pattern matters.

Hypothesis: detectability is determined by the effective rank of the update
ΔW_fc = W_fc_local - W_fc_global in the subspace visible to probe images.

Three measurements per (trigger, client, seed):

  1. delta_W_eff_rank
     Effective rank of ΔW_fc (10×512 linear layer update) via SVD participation
     ratio: exp(−Σ p_i log p_i) where p_i = σ_i² / Σσ_j².
     Range [1, 10]. Low = update concentrated in a few singular directions.

  2. spatial_entropy
     Entropy of the per-pixel gradient magnitude map: g(h,w) = ||∂P(target)/∂x[:,h,w]||.
     Measures how diffuse the model's spatial sensitivity is.
     Range [0, log(H×W)]. Low = concentrated at specific positions (patch-like).

  3. probe_detectable_mass
     Fraction of ΔW_fc's spectral energy whose singular vectors are aligned with
     the GAN probe feature centroid:
       aligned_i = (cos(probe_centroid, V_i) > cos_threshold)
       detectable_mass = Σ_{aligned_i} σ_i² / Σ σ_j²
     High = much of the weight update points into probe-reachable feature space.
     Low = weight update is in directions probes can't activate.

Prediction:
  - patch, cov_1024 (detected):    low delta_W_eff_rank OR high probe_detectable_mass
  - cov_32–512, checker (evades):  high delta_W_eff_rank AND low probe_detectable_mass
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
ap.add_argument('--n-ref',       type=int, default=50,
                help='Reference images for gradient computation')
ap.add_argument('--cos-thresh',  type=float, default=0.1,
                help='Cosine threshold for probe_detectable_mass')
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
W_fc_global = global_state['linear1.weight'].to(device)   # (10, 512)
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)

# Reference images for gradient computation (clean non-target images)
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs).to(device)
all_lbls = torch.cat(all_lbls).to(device)
ref_imgs = all_imgs[all_lbls != TARGET][:args.n_ref]
print(f'Reference images for gradient: {ref_imgs.shape}', flush=True)

C, H, W = 3, 32, 32
_cpu = torch.device('cpu')
_patch_mask_t  = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)
_patch_mask_np = _patch_mask_t.numpy().astype(bool)

# ── GAN probe centroid through global model (once) ────────────────────────────
def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)

g_model = ResNet8(num_classes=10).to(device)
g_model.load_state_dict(global_state)
g_model.eval()
torch.manual_seed(42); np.random.seed(42)
probe_imgs_g = gan_defense(global_state, global_state, params, the_epoch=1)[0]
with torch.no_grad():
    probe_centroid_g = features(g_model, probe_imgs_g).mean(0)  # (512,)
print('Probe centroid computed.', flush=True)

# ── Trigger patterns (matching check16) ──────────────────────────────────────
COVERAGE_LEVELS = [16, 32, 64, 128, 256, 512, 1024]

def make_coverage_pattern(n_pixels, mask_seed):
    rng = np.random.RandomState(mask_seed)
    spatial = rng.choice(H * W, size=n_pixels, replace=False)
    pat = np.zeros((C, H, W), dtype=np.float32)
    for pos in spatial:
        h_pos, w_pos = divmod(int(pos), W)
        pat[:, h_pos, w_pos] = args.amplitude
    return pat

coverage_patterns = {n: make_coverage_pattern(n, idx)
                     for idx, n in enumerate(COVERAGE_LEVELS)}

rows_c = np.arange(H).reshape(H, 1)
cols_c = np.arange(W).reshape(1, W)
checker_hw = ((rows_c + cols_c) % 2).astype(np.float32) * 2.0 - 1.0
checker_pat = np.stack([checker_hw * args.amplitude] * C, axis=0)

_patch_img = np.zeros((C, H, W), dtype=np.float32)
_patch_img[_patch_mask_np] = 1.0

trigger_list = (
    [('patch',   16,   _patch_img)] +
    [(f'cov_{n}', n, coverage_patterns[n]) for n in COVERAGE_LEVELS] +
    [('checker', 1024, checker_pat)]
)

# ── Rank measurement functions ─────────────────────────────────────────────────

def delta_W_eff_rank(w_local):
    """Effective rank of ΔW_fc via SVD participation ratio."""
    W_fc_local = w_local['linear1.weight'].to(device)  # (10, 512)
    delta = W_fc_local - W_fc_global                   # (10, 512)
    S = torch.linalg.svdvals(delta)                    # (min(10,512),) = (10,)
    p = S**2 / (S**2).sum()
    eff_rank = torch.exp(-(p * torch.log(p + 1e-12)).sum()).item()
    return eff_rank


def spatial_entropy(w_local):
    """Entropy of per-pixel gradient magnitude of P(target) averaged over ref images."""
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(w_local)
    m.eval()
    grad_accum = torch.zeros(H, W, device=device)
    for i in range(0, len(ref_imgs), 10):
        batch = ref_imgs[i:i+10].clone().requires_grad_(True)
        logits = m(batch)
        target_prob = torch.softmax(logits, 1)[:, TARGET].sum()
        target_prob.backward()
        g = batch.grad.detach().abs().mean(1)   # (B, H, W) — mean over channels
        grad_accum += g.sum(0)
    grad_accum = grad_accum / grad_accum.sum()
    entropy = -(grad_accum * torch.log(grad_accum + 1e-12)).sum().item()
    return entropy


def probe_detectable_mass(w_local):
    """Fraction of ΔW_fc energy whose right singular vectors align with probe centroid."""
    W_fc_local = w_local['linear1.weight'].to(device)
    delta = W_fc_local - W_fc_global                   # (10, 512)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)  # Vh: (10, 512)
    # Each row of Vh is a right singular vector in R^512
    cos_sims = F.cosine_similarity(Vh, probe_centroid_g.unsqueeze(0))  # (10,)
    aligned = (cos_sims.abs() > args.cos_thresh)
    energy_total   = (S**2).sum()
    energy_aligned = (S[aligned]**2).sum()
    return (energy_aligned / (energy_total + 1e-12)).item()


# ── Also re-run s3 / feat_align for sanity check ─────────────────────────────
def measure_s3(w_local):
    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(w_local)
    m.eval()
    W_l_target = w_local['linear1.weight'].to(device)[TARGET]
    torch.manual_seed(42); np.random.seed(42)
    probe_imgs = gan_defense(global_state, w_local, params, the_epoch=1)[0]
    with torch.no_grad():
        probs = torch.softmax(m(probe_imgs), 1)
        s3 = probs.mean(0)[TARGET].item()
        feats = features(m, probe_imgs)
    feat_align = F.cosine_similarity(
        feats.mean(0, keepdim=True), W_l_target.unsqueeze(0)).item()
    return s3, feat_align


# ── Main loop ─────────────────────────────────────────────────────────────────
COLS = ['trigger', 'n_pixels', 'client', 'seed',
        'delta_W_eff_rank', 'spatial_entropy', 'probe_detectable_mass',
        's3', 'feat_align']

rows = []
print(f'\n{"trigger":>12}  {"n_px":>6}  {"client":>6}  {"seed":>4}  '
      f'{"ΔW_rank":>8}  {"sp_ent":>7}  {"det_mass":>9}  {"s3":>7}  {"feat_al":>8}')
print('-' * 80)


def train_local(trigger_name, pat_np, client_id, seed):
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
    return w_local


for tname, n_px, pat in trigger_list:
    for client_id in args.clients:
        for seed in args.seeds:
            w_local = train_local(tname, pat, client_id, seed)

            dw_rank = delta_W_eff_rank(w_local)
            sp_ent  = spatial_entropy(w_local)
            det_mass = probe_detectable_mass(w_local)
            s3, feat_al = measure_s3(w_local)

            row = {'trigger': tname, 'n_pixels': n_px,
                   'client': client_id, 'seed': seed,
                   'delta_W_eff_rank': dw_rank,
                   'spatial_entropy': sp_ent,
                   'probe_detectable_mass': det_mass,
                   's3': s3, 'feat_align': feat_al}
            rows.append(row)
            print(f'{tname:>12}  {n_px:>6}  {client_id:>6}  {seed:>4}  '
                  f'{dw_rank:>8.3f}  {sp_ent:>7.3f}  {det_mass:>9.4f}  '
                  f'{s3:>7.4f}  {feat_al:>8.4f}', flush=True)


# ── Summary ───────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst)/len(lst) if lst else float('nan')

print('\n\n=== Mean over clients & seeds ===')
print(f"{'trigger':>12}  {'n_px':>6}  {'ΔW_rank':>8}  "
      f"{'sp_ent':>7}  {'det_mass':>9}  {'s3':>7}  {'feat_al':>8}")
print('-' * 72)
for tname, n_px, _ in trigger_list:
    tr = [r for r in rows if r['trigger'] == tname]
    print(f"{tname:>12}  {n_px:>6}  "
          f"{mn([r['delta_W_eff_rank']        for r in tr]):>8.3f}  "
          f"{mn([r['spatial_entropy']          for r in tr]):>7.3f}  "
          f"{mn([r['probe_detectable_mass']    for r in tr]):>9.4f}  "
          f"{mn([r['s3']                       for r in tr]):>7.4f}  "
          f"{mn([r['feat_align']               for r in tr]):>8.4f}")

print('\n=== Correlations with s3 ===')
import statistics
all_s3 = [r['s3'] for r in rows]
for col in ['delta_W_eff_rank', 'spatial_entropy', 'probe_detectable_mass', 'feat_align']:
    vals = [r[col] for r in rows]
    n = len(vals)
    mx = statistics.mean(vals); my = statistics.mean(all_s3)
    num = sum((x-mx)*(y-my) for x, y in zip(vals, all_s3))
    denom = (sum((x-mx)**2 for x in vals) * sum((y-my)**2 for y in all_s3))**0.5
    corr = num/denom if denom > 1e-12 else 0.0
    print(f'  cor({col}, s3) = {corr:+.4f}')

print('\n=== Key comparison: cov_1024 vs checker (same coverage, different detection) ===')
for tname in ['cov_1024', 'checker']:
    tr = [r for r in rows if r['trigger'] == tname]
    print(f"  {tname:>10}: ΔW_rank={mn([r['delta_W_eff_rank'] for r in tr]):.3f}  "
          f"det_mass={mn([r['probe_detectable_mass'] for r in tr]):.4f}  "
          f"s3={mn([r['s3'] for r in tr]):.4f}")

csv_path = os.path.join(args.out, 'check17_rank_analysis.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(rows)
print(f'\nSaved to {csv_path}')
