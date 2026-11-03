"""
Measure three terms of the backdoor implantation dynamics:

  dm/dt = eta_eff * (g_adv . v_m) - lambda * (m - m_nat)

  m_nat      : target margin on triggered inputs under clean FL model (direct)
  g_adv.v_m  : projection of one adversarial local-training step onto margin gradient
  lambda     : benign restoration rate (perturb clean model, run k benign rounds, fit decay)

  Implantation prediction: m* = m_nat + eta_eff*(g_adv.v_m)/lambda > 0

Compares checker (YES), blend_87 (NO), blend_75 (NO).
If ratio (g_adv.v_m)/lambda separates checker from blends where max_col didn't fully,
we have the mechanism. If ratio also fails -> model is incomplete.

Output: ~/ffl_step5/step5_dynamics_measure.txt
"""

import sys, os, pickle, copy, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_PATH    = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_fl_clean_model.pt'
TRAIN_DIR    = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test/data/cifar-10-batches-py'
OUT_PATH     = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_dynamics_measure.txt'
AMPLITUDE    = 0.1145
TARGET_CLS   = 0          # airplane
N_CLIENTS    = 100
N_ROUND      = 10         # clients per round
ALPHA_DIR    = 0.5
LOCAL_EPOCHS = 2
LR           = 0.01
MOMENTUM     = 0.9
BATCH        = 64
K_BENIGN     = 30         # benign rounds for lambda estimation
N_TEST_TRIG  = 500
PART_SEED    = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}", flush=True)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)
std_t = torch.tensor(STD).to(device)

# ── Load CIFAR-10 ─────────────────────────────────────────────────────────────
def load_cifar10(data_dir, split='train'):
    fnames = ([f'data_batch_{i}' for i in range(1,6)] if split=='train'
              else ['test_batch'])
    imgs, labels = [], []
    for fn in fnames:
        with open(os.path.join(data_dir, fn), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        imgs.append(d[b'data'].reshape(-1,3,32,32).astype(np.float32)/255.0)
        labels.extend(d[b'labels'])
    imgs_norm = (np.concatenate(imgs, axis=0) - MEAN) / STD
    return imgs_norm, np.array(labels)

print("Loading CIFAR-10...", flush=True)
train_imgs, train_labels = load_cifar10(TRAIN_DIR, 'train')
test_imgs,  test_labels  = load_cifar10(TRAIN_DIR, 'test')
test_imgs_t = torch.tensor(test_imgs[:N_TEST_TRIG]).to(device)

# ── Dirichlet partition ───────────────────────────────────────────────────────
def dirichlet_partition(labels, n_clients, alpha, seed=1):
    rng   = np.random.RandomState(seed)
    n_cls = int(labels.max()) + 1
    cids  = [[] for _ in range(n_clients)]
    for c in range(n_cls):
        idx  = np.where(labels == c)[0]; rng.shuffle(idx)
        prop = rng.dirichlet([alpha]*n_clients)
        sp   = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, sp)):
            cids[cid].extend(chunk.tolist())
    return cids

print("Partitioning (Dirichlet α=0.5)...", flush=True)
client_idxs = dirichlet_partition(train_labels, N_CLIENTS, ALPHA_DIR, PART_SEED)

# ── Model helpers ─────────────────────────────────────────────────────────────
def make_model():
    return ResNet8(num_classes=10).to(device)

def params_to_flat(model):
    """Flatten model.parameters() in order — consistent across all functions."""
    return torch.cat([p.float().flatten() for p in model.parameters()])

def apply_flat_to_model(flat, model):
    """Write flat parameter vector back into model.parameters() in order."""
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[idx:idx+n].reshape(p.shape).to(p.dtype))
            idx += n

clean_ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
clean_sd   = clean_ckpt['global_state']
print(f"Clean FL model loaded (round {clean_ckpt['round']}).", flush=True)

def load_clean():
    m = make_model()
    m.load_state_dict(clean_sd)
    return m

# ── Trigger application ───────────────────────────────────────────────────────
def apply_trigger(imgs_t, delta_raw_np):
    delta_t = torch.tensor(delta_raw_np, dtype=torch.float32).to(device) / std_t
    return torch.clamp(imgs_t + delta_t.unsqueeze(0), -4.0, 4.0)

# ── Target margin measurement ─────────────────────────────────────────────────
_mask = torch.ones(10, dtype=torch.bool)
_mask[TARGET_CLS] = False

def target_margin(sd, imgs_trig_t):
    """Mean target margin: logit[target] - max_{c!=target} logit[c]."""
    m = make_model(); m.load_state_dict(sd); m.eval()
    all_m = []
    with torch.no_grad():
        for i in range(0, len(imgs_trig_t), 256):
            logits = m(imgs_trig_t[i:i+256])
            lt     = logits[:, TARGET_CLS]
            lmax   = logits[:, _mask].max(dim=1).values
            all_m.append((lt - lmax).cpu())
    return float(torch.cat(all_m).mean())

# ── Margin gradient v_m (grad of mean margin w.r.t. all params) ───────────────
def margin_gradient_flat(sd, imgs_trig_t):
    """Gradient of mean target margin w.r.t. model.parameters() — same order as params_to_flat."""
    m = make_model(); m.load_state_dict(sd); m.train(); m.zero_grad()
    n_batches = 0
    for i in range(0, len(imgs_trig_t), 256):
        logits = m(imgs_trig_t[i:i+256])
        lt   = logits[:, TARGET_CLS]
        lmax = logits[:, _mask].max(dim=1).values
        (lt - lmax).mean().backward()
        n_batches += 1
    grad_flat = torch.cat([p.grad.float().flatten() for p in m.parameters()])
    return grad_flat.detach() / max(1, n_batches)

# ── Adversarial weight update ─────────────────────────────────────────────────
def adv_weight_update(sd, delta_raw_np):
    """
    One adversarial client: local train on triggered data (target label).
    Returns flattened parameter delta (θ_after - θ_before) via model.parameters() order.
    """
    m = make_model(); m.load_state_dict(copy.deepcopy(sd)); m.train()
    before_flat = params_to_flat(m).clone()
    opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    imgs_trig = apply_trigger(test_imgs_t, delta_raw_np)
    targets_t = torch.full((len(imgs_trig),), TARGET_CLS, dtype=torch.long).to(device)
    for _ in range(LOCAL_EPOCHS):
        perm = torch.randperm(len(imgs_trig))
        for i in range(0, len(imgs_trig), BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            criterion(m(imgs_trig[idx]), targets_t[idx]).backward()
            opt.step()
    after_flat = params_to_flat(m)
    return (after_flat - before_flat).detach()

# ── One benign FedAvg round ───────────────────────────────────────────────────
def benign_fedavg_round(global_sd, rng):
    """One round of benign-only FedAvg. Returns new state dict."""
    selected = rng.choice(N_CLIENTS, N_ROUND, replace=False)
    float_keys = [k for k, v in global_sd.items() if v.dtype != torch.int64]
    deltas, sizes = [], []
    for cid in selected:
        idxs = client_idxs[cid]
        if len(idxs) < 8:
            continue
        loc = make_model(); loc.load_state_dict(copy.deepcopy(global_sd)); loc.train()
        opt = torch.optim.SGD(loc.parameters(), lr=LR, momentum=MOMENTUM)
        crit = nn.CrossEntropyLoss()
        xi = torch.tensor(train_imgs[idxs]).to(device)
        yi = torch.tensor(train_labels[idxs], dtype=torch.long).to(device)
        for _ in range(LOCAL_EPOCHS):
            perm = torch.randperm(len(xi))
            for i in range(0, len(xi), BATCH):
                b = perm[i:i+BATCH]
                opt.zero_grad()
                crit(loc(xi[b]), yi[b]).backward()
                opt.step()
        d = {k: (loc.state_dict()[k].float() - global_sd[k].float()).detach()
             for k in float_keys}
        deltas.append(d); sizes.append(len(idxs))
    if not deltas:
        return global_sd
    total = sum(sizes)
    new_sd = copy.deepcopy(global_sd)
    for k in float_keys:
        agg = sum(deltas[i][k] * (sizes[i]/total) for i in range(len(deltas)))
        new_sd[k] = (global_sd[k].float() + agg).to(global_sd[k].dtype)
    return new_sd

# ── Main ──────────────────────────────────────────────────────────────────────
TRIGGERS = {
    'checker'     : (make_raw_delta('checker',      AMPLITUDE), 'YES'),
    'blend_87'    : (make_raw_delta('blend_87',     AMPLITUDE), 'NO'),
    'blend_75'    : (make_raw_delta('blend_75',     AMPLITUDE), 'NO'),
    'low_col_opt' : (make_raw_delta('low_col_opt',  AMPLITUDE), 'NO'),
}

results = {}
lines   = [
    "=== Backdoor implantation dynamics: three-term decomposition ===",
    f"  FL clean model, FedAvg α=0.5 | K_benign={K_BENIGN} | target=airplane",
    "",
]

for trig_name, (delta_raw, known) in TRIGGERS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"Trigger: {trig_name}  (known={known})", flush=True)

    imgs_trig_t = apply_trigger(test_imgs_t, delta_raw)

    # Term 1: m_nat
    m_nat = target_margin(clean_sd, imgs_trig_t)
    print(f"  m_nat = {m_nat:+.4f}", flush=True)

    # Term 2: g_adv . v_m
    print("  Computing g_adv (adversarial local train)...", flush=True)
    g_adv = adv_weight_update(clean_sd, delta_raw)
    g_norm = float(g_adv.norm())
    g_adv_unit = g_adv / (g_adv.norm() + 1e-9)
    print(f"  ||g_adv|| = {g_norm:.4f}", flush=True)

    print("  Computing v_m (margin gradient)...", flush=True)
    v_m = margin_gradient_flat(clean_sd, imgs_trig_t)
    v_m_unit = v_m / (v_m.norm() + 1e-9)
    proj = float(torch.dot(g_adv_unit, v_m_unit))
    print(f"  g_adv . v_m (normalised cosine) = {proj:+.4f}", flush=True)

    # Term 3: lambda
    # Perturb clean model by one adversarial step, run benign rounds, fit decay
    eps = g_norm   # perturbation = one adv-step magnitude
    m_perturb = load_clean()
    clean_flat = params_to_flat(m_perturb)
    apply_flat_to_model(clean_flat + eps * g_adv_unit, m_perturb)
    perturbed_sd = m_perturb.state_dict()

    m0 = target_margin(perturbed_sd, imgs_trig_t)
    print(f"  m_0 (perturbed) = {m0:+.4f}  m_nat = {m_nat:+.4f}", flush=True)
    print(f"  Running {K_BENIGN} benign rounds...", flush=True)

    rng_b   = np.random.RandomState(99)
    margins = [m0]
    cur_sd  = perturbed_sd

    for rnd in range(K_BENIGN):
        cur_sd = benign_fedavg_round(cur_sd, rng_b)
        m_t    = target_margin(cur_sd, imgs_trig_t)
        margins.append(m_t)
        if (rnd+1) % 5 == 0:
            print(f"    round {rnd+1:3d}: m = {m_t:+.4f}", flush=True)

    margins = np.array(margins)

    # Fit decay: m_t = m_nat + (m0-m_nat)*exp(-lambda*t)
    denom = m0 - m_nat
    if abs(denom) > 1e-6:
        ratios = np.clip((margins - m_nat) / denom, 1e-9, None)
        log_r  = np.log(ratios)
        t_vals = np.arange(len(margins))
        finite = np.isfinite(log_r)
        lam_fit = float(-np.polyfit(t_vals[finite], log_r[finite], 1)[0]) if finite.sum()>2 else float('nan')
    else:
        lam_fit = float('nan')

    # Half-life
    m_half = m_nat + 0.5 * denom
    hl_idx = next((i for i,m in enumerate(margins)
                   if (denom > 0 and m <= m_half) or (denom < 0 and m >= m_half)),
                  K_BENIGN)
    print(f"  lambda_fit = {lam_fit:.4f}  half_life = {hl_idx} rounds", flush=True)

    ratio = proj / lam_fit if (not np.isnan(lam_fit) and abs(lam_fit) > 1e-9) else float('nan')
    print(f"  (g_adv.v_m)/lambda = {ratio:+.4f}  |m_nat| = {abs(m_nat):.4f}", flush=True)

    results[trig_name] = {
        'm_nat': m_nat, 'g_adv_norm': g_norm, 'proj': proj,
        'lambda': lam_fit, 'half_life': hl_idx, 'ratio': ratio,
        'margins': margins.tolist(), 'known': known,
    }

    lines += [
        f"=== {trig_name} (implants={known}) ===",
        f"  m_nat                    = {m_nat:+.4f}",
        f"  ||g_adv||                = {g_norm:.4f}",
        f"  g_adv . v_m (cos, norm.) = {proj:+.4f}",
        f"  lambda_fit               = {lam_fit:.4f}  (half-life: {hl_idx} rounds)",
        f"  ratio (g.v_m)/lambda     = {ratio:+.4f}  |m_nat| = {abs(m_nat):.4f}",
        f"  benign trajectory (every 5 rounds): " +
        ", ".join(f"r{i*5}:{m:.3f}" for i,m in enumerate(margins[::5])),
        "",
    ]

# Summary table
lines += [
    "=== Summary table ===",
    f"  {'trigger':10s}  {'known':>6s}  {'m_nat':>7s}  {'g.v_m':>7s}  "
    f"{'lambda':>7s}  {'ratio':>7s}  {'half_life':>10s}",
    f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}",
]
for trig_name, r in results.items():
    lines.append(
        f"  {trig_name:10s}  {r['known']:>6s}  {r['m_nat']:+7.4f}  {r['proj']:+7.4f}  "
        f"{r['lambda']:7.4f}  {r['ratio']:+7.4f}  {r['half_life']:10d}"
    )
lines += [
    "",
    "Interpretation:",
    "  Implants iff m_nat + eta_eff * ratio > 0.",
    "  If ratio(checker) >> ratio(blend_87): lambda or g.v_m is the separating term.",
    "  If ratio fails to separate: model incomplete (missing term — e.g. defence-specific).",
]

out = '\n'.join(lines)
print("\n" + out, flush=True)
with open(OUT_PATH, 'w') as f:
    f.write(out + '\n')
print(f"\nSaved: {OUT_PATH}")
