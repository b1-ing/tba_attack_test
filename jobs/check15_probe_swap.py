#!/usr/bin/env python3
"""Check 15: Swap the probe generator — GAN vs real server data.

Check 14 showed that global triggers evade Stage-3 detection (s3≈0.03) while local
triggers are detected (s3≈0.14–0.24). The proposed mechanism is that GAN probes are
natural images that share no feature-space overlap with globally synthetic patterns.

Competing hypotheses:
  H_GAN:  The GAN's natural-image prior is specifically the problem.  Using real
          held-out images as probes would detect global triggers.
  H_NAT:  Any natural-image probe fails against global triggers.  The blindspot is
          structural — not GAN-specific — because global patterns are orthogonal to
          the natural image manifold regardless of how the probes are generated.

Experiment:
  Run the same 4 triggers from check14, but measure feat_align and s3 under TWO
  probe regimes for each (client, trigger, seed) triple:
    probe_gan   — FilterFL's own GAN (baseline, replicates check14)
    probe_real  — N_PROBE random test images, mixed classes (server-data style)
    probe_cls0  — N_PROBE test images from target class only (strongest real signal)

If H_GAN:  probe_real / probe_cls0 should detect global triggers (high s3).
If H_NAT:  all three probe types give s3≈0 for global triggers.

Additional measurement (direct manifold distance):
  For each trigger pattern, compute cosine distance between:
    (a) trigger_features  — features of the trigger pattern itself through global model
    (b) probe_centroid    — mean features of GAN probes through global model
  This directly tests whether the trigger lies "outside the probe manifold".
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',    default='FilterFL')
ap.add_argument('--warmup-dir',  default=None)
ap.add_argument('--out',         default='.')
ap.add_argument('--clients',     type=int, nargs='+', default=[34, 36, 47, 60])
ap.add_argument('--seeds',       type=int, nargs='+', default=[1, 2, 3])
ap.add_argument('--n-test',      type=int, default=500)
ap.add_argument('--n-probe',     type=int, default=200,
                help='Number of probe images for real-data regimes')
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

# ── Collect full test set ─────────────────────────────────────────────────────
all_imgs, all_lbls = [], []
for imgs, lbls in handle.test_data:
    all_imgs.append(imgs); all_lbls.append(lbls)
all_imgs = torch.cat(all_imgs).to(device)
all_lbls = torch.cat(all_lbls).to(device)
print(f'Full test set: {all_imgs.shape}', flush=True)

# ASR evaluation pool: up to n_test non-target images
non_target_mask = (all_lbls != TARGET)
test_imgs = all_imgs[non_target_mask][:args.n_test]
print(f'ASR eval pool (non-target): {test_imgs.shape}', flush=True)

# Real probe pools
rng_probe = np.random.RandomState(99)
mixed_idx = rng_probe.choice(len(all_imgs), size=args.n_probe, replace=False)
probe_real_mixed = all_imgs[mixed_idx]                         # all classes

target_mask = (all_lbls == TARGET)
target_imgs = all_imgs[target_mask]
probe_real_cls0 = target_imgs[:args.n_probe]                   # target class only
print(f'Real probes — mixed: {probe_real_mixed.shape}, '
      f'cls0: {probe_real_cls0.shape}', flush=True)

C, H, W = 3, 32, 32

# ── Trigger patterns (identical to check14) ───────────────────────────────────
_cpu = torch.device('cpu')
_patch_mask_t  = train_mod.get_pattern_mask(params, 0, (C, H, W), _cpu)
_patch_mask_np = _patch_mask_t.numpy().astype(bool)

def _additive_local_pattern(amplitude):
    pat = np.zeros((C, H, W), dtype=np.float32)
    pat[_patch_mask_np] = amplitude
    return pat

def _replace_global_pattern():
    k_x = 10
    col_idx = np.arange(W, dtype=np.float32)
    stripe = 0.5 * np.cos(2 * np.pi * k_x * col_idx / W)   # zero-mean, ±0.5
    pattern = np.stack([np.tile(stripe, (H, 1))] * C, axis=0)
    return pattern.astype(np.float32)

_additive_local_05  = _additive_local_pattern(0.5)
_additive_local_10  = _additive_local_pattern(1.0)
_replace_global_pat = _replace_global_pattern()

# ── Feature extractor ─────────────────────────────────────────────────────────
def features(model, x):
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)

# ── Direct manifold distance (global model, done once) ────────────────────────
print('\n=== Manifold distance: trigger features vs GAN probe centroid (global model) ===')
torch.manual_seed(42); np.random.seed(42)
g_model = ResNet8(num_classes=10).to(device)
g_model.load_state_dict(global_state)
g_model.eval()

probe_imgs_global_gan = gan_defense(global_state, global_state, params, the_epoch=1)[0]
with torch.no_grad():
    probe_feats_global = features(g_model, probe_imgs_global_gan)
    probe_centroid_global = probe_feats_global.mean(0)      # (512,)
    probe_real_feats = features(g_model, probe_real_mixed)
    probe_real_centroid = probe_real_feats.mean(0)

def manifold_dist(pattern_np, label):
    """Cosine distance from trigger pattern (as image batch) to probe centroid."""
    pat_t = torch.from_numpy(pattern_np).to(device).unsqueeze(0)  # (1,C,H,W)
    with torch.no_grad():
        feat = features(g_model, pat_t).squeeze(0)   # (512,)
    cos_gan  = F.cosine_similarity(feat.unsqueeze(0), probe_centroid_global.unsqueeze(0)).item()
    cos_real = F.cosine_similarity(feat.unsqueeze(0), probe_real_centroid.unsqueeze(0)).item()
    print(f'  {label:>22}: cos(trigger_feat, GAN_centroid)={cos_gan:+.4f}  '
          f'cos(trigger_feat, real_centroid)={cos_real:+.4f}')

# patch: use a 32x32 image with patch pixels set to 1, rest to 0
_patch_img = np.zeros((C, H, W), dtype=np.float32)
_patch_img[_patch_mask_np] = 1.0
manifold_dist(_patch_img,          'patch_delta')
manifold_dist(_additive_local_05,  'additive_local_05')
manifold_dist(_replace_global_pat, 'replace_global')

# ── Training + measurement ────────────────────────────────────────────────────
TRIGGERS = ['patch', 'additive_local_05', 'additive_local_hi', 'replace_global']
PROBES   = ['gan', 'real_mixed', 'real_cls0']
COLS = ['trigger', 'client', 'seed', 'probe_type',
        'feat_align', 's3', 'asr_proxy']

rows = []
hdr = f"{'trigger':>20}  {'client':>6}  {'seed':>4}  {'probe':>10}  " \
      f"{'feat_al':>8}  {'s3':>7}  {'asr_px':>8}"
print(f'\n{hdr}')
print('-' * len(hdr))


def measure_one(client_id, trigger_name, seed):
    """Train local model once, then evaluate under all three probe regimes."""
    torch.manual_seed(seed); np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate'] = 0.50

    import train as tm
    _orig = tm.apply_pattern_batch_gpu

    if trigger_name == 'patch':
        def patched(images, pms, index, handle_, poison_mask):
            return _orig(images, pms, index, handle_, poison_mask)

    elif trigger_name == 'additive_local_05':
        pat_np = _additive_local_05
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            poisoned = images + pat_t.unsqueeze(0)
            labels = torch.full((images.size(0),), TARGET,
                                device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), labels

    elif trigger_name == 'additive_local_hi':
        pat_np = _additive_local_10
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            poisoned = torch.clamp(images + pat_t.unsqueeze(0), -1.0, 1.0)
            labels = torch.full((images.size(0),), TARGET,
                                device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), poisoned, images), labels

    elif trigger_name == 'replace_global':
        pat_np = _replace_global_pat
        def patched(images, pms, index, handle_, poison_mask):
            pm_exp = poison_mask.view(-1, 1, 1, 1)
            pat_t = torch.from_numpy(pat_np).to(images.device)
            replaced = pat_t.unsqueeze(0).expand_as(images)
            labels = torch.full((images.size(0),), TARGET,
                                device=images.device, dtype=torch.long)
            return torch.where(pm_exp.expand_as(images).bool(), replaced, images), labels

    else:
        raise ValueError(trigger_name)

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

    # ASR proxy (trigger applied to non-target test images)
    with torch.no_grad():
        if trigger_name == 'patch':
            mask_b = _patch_mask_t.to(device).bool().unsqueeze(0).expand_as(test_imgs)
            triggered = torch.where(mask_b,
                                    torch.tensor(1.0, device=device), test_imgs)
        elif trigger_name == 'additive_local_05':
            triggered = test_imgs + torch.from_numpy(_additive_local_05).to(device)
        elif trigger_name == 'additive_local_hi':
            triggered = torch.clamp(
                test_imgs + torch.from_numpy(_additive_local_10).to(device), -1, 1)
        elif trigger_name == 'replace_global':
            triggered = torch.from_numpy(_replace_global_pat).to(device)\
                            .unsqueeze(0).expand_as(test_imgs).clone()
        asr_proxy = torch.softmax(m(triggered), 1).mean(0)[TARGET].item()

    # Probe sweep
    results = {}
    torch.manual_seed(42); np.random.seed(42)
    gan_imgs = gan_defense(global_state, w_local, params, the_epoch=1)[0]

    for probe_name, probe_imgs in [
        ('gan',       gan_imgs),
        ('real_mixed', probe_real_mixed[:args.n_probe]),
        ('real_cls0',  probe_real_cls0[:args.n_probe]),
    ]:
        with torch.no_grad():
            probs = torch.softmax(m(probe_imgs), 1)
            s3 = probs.mean(0)[TARGET].item()
            feats = features(m, probe_imgs)
        feat_align = F.cosine_similarity(
            feats.mean(0, keepdim=True),
            W_l_target.unsqueeze(0)).item()
        results[probe_name] = {'feat_align': feat_align, 's3': s3,
                               'asr_proxy': asr_proxy}

    return results


for trigger_name in TRIGGERS:
    for client_id in args.clients:
        for seed in args.seeds:
            res = measure_one(client_id, trigger_name, seed)
            for probe_name in PROBES:
                r = res[probe_name]
                row = {'trigger': trigger_name, 'client': client_id,
                       'seed': seed, 'probe_type': probe_name, **r}
                rows.append(row)
                print(f"{trigger_name:>20}  {client_id:>6}  {seed:>4}  "
                      f"{probe_name:>10}  {r['feat_align']:>8.4f}  "
                      f"{r['s3']:>7.4f}  {r['asr_proxy']:>8.4f}",
                      flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
def mn(lst): return sum(lst)/len(lst) if lst else float('nan')

print('\n\n=== Mean s3 by trigger × probe type ===')
print(f"{'trigger':>20}  {'gan':>8}  {'real_mix':>9}  {'real_cls0':>10}  {'asr_px':>8}")
print('-' * 65)
for t in TRIGGERS:
    s3_gan   = mn([r['s3'] for r in rows if r['trigger']==t and r['probe_type']=='gan'])
    s3_rmix  = mn([r['s3'] for r in rows if r['trigger']==t and r['probe_type']=='real_mixed'])
    s3_rcls  = mn([r['s3'] for r in rows if r['trigger']==t and r['probe_type']=='real_cls0'])
    asr      = mn([r['asr_proxy'] for r in rows if r['trigger']==t and r['probe_type']=='gan'])
    print(f"{t:>20}  {s3_gan:>8.4f}  {s3_rmix:>9.4f}  {s3_rcls:>10.4f}  {asr:>8.4f}")

print('\n=== Interpretation ===')
rg_gan  = mn([r['s3'] for r in rows if r['trigger']=='replace_global' and r['probe_type']=='gan'])
rg_mix  = mn([r['s3'] for r in rows if r['trigger']=='replace_global' and r['probe_type']=='real_mixed'])
rg_cls  = mn([r['s3'] for r in rows if r['trigger']=='replace_global' and r['probe_type']=='real_cls0'])
thresh = 0.05

if rg_mix > thresh or rg_cls > thresh:
    print(f'  H_GAN supported: real probes detect global trigger '
          f'(real_mixed s3={rg_mix:.4f}, real_cls0 s3={rg_cls:.4f} vs GAN s3={rg_gan:.4f})')
    print('  → the GAN\'s natural-image prior is specifically the problem; '
          'server-data probes would fix the defense')
else:
    print(f'  H_NAT supported: ALL probe types fail against global trigger '
          f'(gan={rg_gan:.4f}, real_mixed={rg_mix:.4f}, real_cls0={rg_cls:.4f})')
    print('  → blindspot is structural; any natural-image probe fails; '
          'fixing the GAN alone is insufficient')

csv_path = os.path.join(args.out, 'check15_probe_swap.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader(); wr.writerows(rows)
print(f'\nSaved to {csv_path}')
