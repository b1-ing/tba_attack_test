#!/usr/bin/env python3
"""Check 11: Trigger-design as a Stage-3 evasion route.

The proximate driver of Stage-3 is feat_align = cos(mean_probe_features, W_fc[target]).
The GAN (Stage-2) generates smooth probe images via an upsampling CNN generator.
Hypothesis: a trigger whose feature representation is far from what the GAN explores
will produce low feat_align even after backdoor training, evading Stage-3 while
maintaining ASR.

Three trigger designs at rate=0.50 (maximum poisoning):
  'patch'   — current: replace 16 specific pixels with 1.0 (top-row white pixels)
  'checker' — additive pixel-level checkerboard ±0.5 (Nyquist freq, far from smooth manifold)
  'noise'   — additive fixed-seed random noise at L∞=0.5

Key measurements per (client, trigger_type, seed):
  feat_align      — cos(probe_features, W_fc[target]) — Stage-3 detection signal
  trigger_feat_al — cos(triggered_test_features, W_fc[target]) — where the trigger
                    actually sits in feature space (should be high for attack to work)
  s3              — Stage-3 score (p(class 0) on probe images)
  asr_proxy       — p(target class) on triggered clean test images (local model)

The gap between trigger_feat_al and feat_align (probe) is the evasion gap:
  high trigger_feat_al + low feat_align → backdoor lives in a direction the GAN misses.
"""
import argparse, copy, csv, datetime, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code-dir',   default='FilterFL')
ap.add_argument('--warmup-dir', default=None)
ap.add_argument('--out',        default='.')
ap.add_argument('--clients',    type=int,   nargs='+', default=[34, 36])
ap.add_argument('--triggers',               nargs='+',
                default=['patch', 'checker', 'noise'],
                help='Trigger types to sweep')
ap.add_argument('--rate',       type=float, default=0.50)
ap.add_argument('--amplitude',  type=float, default=0.50,
                help='Amplitude for additive triggers (checker/noise)')
ap.add_argument('--seeds',      type=int,   nargs='+', default=[1, 2, 3])
ap.add_argument('--n-test',     type=int,   default=500,
                help='Number of test images for asr_proxy and trigger_feat_al')
ap.add_argument('--alpha',      type=float, default=0.5)
ap.add_argument('--global-seed',type=int,   default=1)
args = ap.parse_args()

code_dir = os.path.abspath(args.code_dir)
sys.path.insert(0, code_dir)
os.chdir(code_dir)
os.makedirs('saved_models', exist_ok=True)
os.makedirs(args.out, exist_ok=True)

import torch
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
assert os.path.exists(ckpt), f'No warmup checkpoint at {ckpt}'
handle.model.load_state_dict(torch.load(ckpt, map_location=device))
print(f'Loaded warmup checkpoint: {ckpt}', flush=True)
global_state = copy.deepcopy(handle.model.state_dict())

# Build a test batch for asr_proxy and trigger_feat_al
print(f'Building test batch (n={args.n_test})...', flush=True)
test_imgs_list, test_lbls_list = [], []
for imgs, lbls in handle.test_data:
    test_imgs_list.append(imgs)
    test_lbls_list.append(lbls)
    if sum(x.size(0) for x in test_imgs_list) >= args.n_test:
        break
test_imgs = torch.cat(test_imgs_list)[:args.n_test].to(device)
test_lbls = torch.cat(test_lbls_list)[:args.n_test].to(device)
print(f'Test batch: {test_imgs.shape}, classes: {test_lbls.unique().tolist()[:10]}',
      flush=True)


def features(model, x):
    """Penultimate layer features (same as check10)."""
    out = torch.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.avgpool(out)
    return torch.flatten(out, 1)


def apply_trigger_to_batch(imgs, trigger_type, p):
    """Apply trigger to a batch of images (for eval, not training)."""
    ttype = p.get('trigger_type', 'patch')
    if ttype in ('checker', 'noise'):
        additive = train_mod._get_additive_trigger(ttype, p, imgs[0].shape, imgs.device)
        return imgs + additive.unsqueeze(0)
    else:
        # patch or inverted: apply via pattern mask
        mask = train_mod.get_pattern_mask(p, 0, imgs[0].shape, imgs.device).bool()
        mask_b = mask.unsqueeze(0).expand_as(imgs)
        val = 1.0 if ttype == 'patch' else 0.0
        return torch.where(mask_b, torch.tensor(val, device=imgs.device, dtype=imgs.dtype), imgs)


def measure(client_id, trigger_type, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    p = copy.deepcopy(params)
    p['poison_rate']      = args.rate
    p['uap_mu']           = 0.0
    p['trigger_type']     = trigger_type
    p['trigger_amplitude']= args.amplitude

    model = ResNet8(num_classes=10).to(device)
    model.load_state_dict(copy.deepcopy(global_state))

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

    W_l = w_local['linear1.weight']  # (10, 512)
    W_l_target = W_l[TARGET]

    # ── Stage-3 score (probe images from GAN) ───────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    g_imgs = gan_defense(global_state, w_local, params, the_epoch=1)
    probe_imgs = g_imgs[0]  # (N, C, H, W)

    m = ResNet8(num_classes=10).to(device)
    m.load_state_dict(w_local)
    m.eval()
    with torch.no_grad():
        logits_probe = m(probe_imgs)
        probs_probe  = torch.softmax(logits_probe, dim=1)
        s3           = probs_probe.mean(0)[TARGET].item()
        feats_probe  = features(m, probe_imgs)       # (N, 512)

    mean_feat_probe = feats_probe.mean(0)            # (512,)
    feat_align = torch.nn.functional.cosine_similarity(
        mean_feat_probe.unsqueeze(0), W_l_target.unsqueeze(0)).item()

    # ── Triggered test images: where does the trigger live in feature space? ─
    with torch.no_grad():
        triggered_test = apply_trigger_to_batch(test_imgs, trigger_type, p)
        feats_trig = features(m, triggered_test)     # (n_test, 512)
        logits_trig = m(triggered_test)
        probs_trig  = torch.softmax(logits_trig, dim=1)
        asr_proxy   = probs_trig.mean(0)[TARGET].item()

    mean_feat_trig = feats_trig.mean(0)              # (512,)
    trigger_feat_al = torch.nn.functional.cosine_similarity(
        mean_feat_trig.unsqueeze(0), W_l_target.unsqueeze(0)).item()

    # ── Clean test images alignment ──────────────────────────────────────────
    with torch.no_grad():
        feats_clean = features(m, test_imgs)         # (n_test, 512)
    mean_feat_clean = feats_clean.mean(0)
    clean_feat_al = torch.nn.functional.cosine_similarity(
        mean_feat_clean.unsqueeze(0), W_l_target.unsqueeze(0)).item()

    # Evasion gap: trigger_feat_al - feat_align(probe)
    # Positive gap → trigger activates target features but probe doesn't find them
    evasion_gap = trigger_feat_al - feat_align

    return {
        'client':         client_id,
        'trigger_type':   trigger_type,
        'rate':           args.rate,
        'seed':           seed,
        'feat_align':     feat_align,        # probe feat vs W_fc[target] — Stage-3 signal
        'trigger_feat_al':trigger_feat_al,   # triggered test feat vs W_fc[target]
        'clean_feat_al':  clean_feat_al,     # clean test feat vs W_fc[target] (baseline)
        'evasion_gap':    evasion_gap,       # trigger_feat_al - feat_align
        's3':             s3,
        'asr_proxy':      asr_proxy,
    }


COLS = ['client', 'trigger_type', 'rate', 'seed',
        'feat_align', 'trigger_feat_al', 'clean_feat_al', 'evasion_gap',
        's3', 'asr_proxy']

rows = []
print(f'\n{"client":>8}  {"trigger":>10}  {"seed":>4}  '
      f'{"feat_al":>8}  {"trig_al":>8}  {"clean_al":>9}  {"gap":>7}  '
      f'{"s3":>7}  {"asr_px":>8}', flush=True)
print('-' * 88)

for client_id in args.clients:
    for trigger_type in args.triggers:
        for seed in args.seeds:
            r = measure(client_id, trigger_type, seed)
            rows.append(r)
            print(f"{r['client']:>8}  {r['trigger_type']:>10}  {r['seed']:>4}  "
                  f"{r['feat_align']:>8.4f}  {r['trigger_feat_al']:>8.4f}  "
                  f"{r['clean_feat_al']:>9.4f}  {r['evasion_gap']:>7.4f}  "
                  f"{r['s3']:>7.4f}  {r['asr_proxy']:>8.4f}", flush=True)


# ── Per-trigger-type summary ─────────────────────────────────────────────────
print('\n\n=== Mean per trigger type (pooled over clients & seeds) ===')
print(f"{'trigger':>12}  {'feat_align':>11}  {'trig_al':>9}  "
      f"{'evasion_gap':>12}  {'s3':>8}  {'asr_proxy':>10}")
print('-' * 70)
for t in args.triggers:
    tr = [r for r in rows if r['trigger_type'] == t]
    def m(k): return np.mean([r[k] for r in tr])
    print(f"{t:>12}  {m('feat_align'):>11.4f}  {m('trigger_feat_al'):>9.4f}  "
          f"{m('evasion_gap'):>12.4f}  {m('s3'):>8.4f}  {m('asr_proxy'):>10.4f}")

# ── Per-client summary ────────────────────────────────────────────────────────
print('\n\n=== Mean per (client, trigger_type) ===')
print(f"{'client':>8}  {'trigger':>10}  {'feat_align':>11}  {'trig_al':>9}  "
      f"{'gap':>7}  {'s3':>8}  {'asr_proxy':>10}")
print('-' * 72)
for c in args.clients:
    for t in args.triggers:
        cr = [r for r in rows if r['client'] == c and r['trigger_type'] == t]
        if not cr: continue
        def m(k): return np.mean([r[k] for r in cr])
        print(f"{c:>8}  {t:>10}  {m('feat_align'):>11.4f}  "
              f"{m('trigger_feat_al'):>9.4f}  {m('evasion_gap'):>7.4f}  "
              f"{m('s3'):>8.4f}  {m('asr_proxy'):>10.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(args.out, 'check11_trigger_design.csv')
with open(csv_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=COLS)
    wr.writeheader()
    wr.writerows(rows)
print(f'\nSaved to {csv_path}')
