import sys, warnings, os, pickle
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')

import numpy as np
import torch

sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta

DATA_DIR  = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test/data/cifar-10-batches-py'
CKPT_PATH = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_fl_clean_model.pt'
OUT_PATH  = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_featcol_fl_results.txt'
AMPLITUDE = 0.1145
N_IMAGES  = 1000
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mean_np = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
std_np  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(3,1,1)

model = ResNet8(num_classes=10).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['global_state'])
model.eval()
print(f"Clean FL model loaded from round {ckpt['round']}")

_penult = {}
model.linear1.register_forward_pre_hook(
    lambda m, inp: _penult.update({'feat': inp[0].detach()})
)

W_fc      = model.linear1.weight.detach().cpu().numpy()
W_fc_norm = W_fc / (np.linalg.norm(W_fc, axis=1, keepdims=True) + 1e-9)
CLASSES   = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_cifar10_test(data_dir):
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    imgs   = d[b'data'].reshape(-1,3,32,32).astype(np.float32) / 255.0
    labels = np.array(d[b'labels'])
    return imgs, labels

imgs_raw, labels = load_cifar10_test(DATA_DIR)
imgs_norm        = (imgs_raw[:N_IMAGES] - mean_np) / std_np

BATCH = 128
def get_feats_logits(imgs_np):
    feats, logits = [], []
    for i in range(0, len(imgs_np), BATCH):
        xb = torch.tensor(imgs_np[i:i+BATCH]).to(device)
        with torch.no_grad():
            out = model(xb)
        feats.append(_penult['feat'].cpu().numpy().copy())
        logits.append(out.cpu().numpy().copy())
    return np.concatenate(feats), np.concatenate(logits)

print("Computing clean features...")
feats_clean, logits_clean = get_feats_logits(imgs_norm)
clean_acc = (logits_clean.argmax(1) == labels[:N_IMAGES]).mean()
print(f"Clean FL model test accuracy (first {N_IMAGES}): {clean_acc:.3f}")

def make_blend(alpha, amp=AMPLITUDE):
    d_chk = make_raw_delta('checker', amp)
    d_str = make_raw_delta('stripes', amp)
    d_mix = alpha * d_chk + (1 - alpha) * d_str
    l2_chk = float(np.linalg.norm(d_chk))
    l2_mix = float(np.linalg.norm(d_mix))
    if l2_mix > 1e-9:
        d_mix = d_mix * (l2_chk / l2_mix)
    return d_mix

def make_stripes_p4(amp=AMPLITUDE):
    C, H, W = 3, 32, 32
    row_sign = ((np.arange(H, dtype=int) // 2) % 2 * 2 - 1).astype(np.float32)
    pat = np.broadcast_to(row_sign.reshape(H,1), (H,W)) * amp
    return np.stack([pat.copy()] * C, axis=0)

named_triggers = {
    'checker'    : make_raw_delta('checker',   AMPLITUDE),
    'blend_87'   : make_blend(0.875),
    'blend_75'   : make_blend(0.75),
    'blend_50'   : make_blend(0.50),
    'blend_25'   : make_blend(0.25),
    'stripes'    : make_raw_delta('stripes',   AMPLITUDE),
    'stripes_p4' : make_stripes_p4(),
    'sin16'      : make_raw_delta('sin16',     AMPLITUDE),
    'corner4x4'  : make_raw_delta('corner4x4', AMPLITUDE),
    'noise'      : make_raw_delta('noise',     AMPLITUDE),
}

known_result = {
    'checker':'YES','blend_87':'?','blend_75':'NO','blend_50':'?','blend_25':'NO',
    'stripes':'NO','stripes_p4':'?','sin16':'NO','corner4x4':'NO','noise':'NO'
}

lines = []
lines.append("=== Feature Collision — FL-trained CLEAN model (FedAvg, alpha=0.5, 150 rounds) ===")
lines.append(f"N={N_IMAGES} test images, amplitude={AMPLITUDE}")
lines.append(f"Clean FL accuracy: {clean_acc:.3f}")
lines.append("")

summary = []
for trig_name, delta_raw in named_triggers.items():
    print(f"Processing: {trig_name}")
    delta_norm = delta_raw / std_np
    imgs_trig  = np.clip(imgs_norm + delta_norm, -4.0, 4.0)
    feats_trig, logits_trig = get_feats_logits(imgs_trig)

    feat_diff  = feats_trig - feats_clean
    feat_dir   = feat_diff.mean(axis=0)
    feat_dir  /= (np.linalg.norm(feat_dir) + 1e-9)
    per_img_norm = np.linalg.norm(feat_diff, axis=1)
    collisions   = W_fc_norm @ feat_dir
    logit_diff   = (logits_trig - logits_clean).mean(axis=0)
    pred_trig    = logits_trig.argmax(1)
    pred_counts  = np.bincount(pred_trig, minlength=10).tolist()

    max_col    = float(np.max(np.abs(collisions)))
    max_cls    = CLASSES[int(np.argmax(np.abs(collisions)))]
    target_col = float(collisions[0])
    dominant   = CLASSES[int(np.argmax(pred_counts))]

    summary.append({
        'name': trig_name, 'max_col': max_col, 'max_cls': max_cls,
        'target_col': target_col, 'feat_norm': float(per_img_norm.mean()),
        'pred_counts': pred_counts, 'dominant': dominant,
        'logit_diff': logit_diff, 'known': known_result[trig_name],
        'collisions': collisions,
    })

    lines.append(f"=== {trig_name} (implant={known_result[trig_name]}) ===")
    lines.append(f"  ||feat_diff|| mean: {per_img_norm.mean():.4f}")
    lines.append(f"  max |collision|: {max_col:.4f}  class={max_cls}")
    lines.append(f"  target col (airplane): {target_col:+.4f}")
    lines.append(f"  dominant pred: {dominant} ({max(pred_counts)}/1000)")
    lines.append(f"  collision: " +
                 ", ".join(f"{CLASSES[c][:4]}:{collisions[c]:+.3f}" for c in range(10)))
    lines.append(f"  logit:     " +
                 ", ".join(f"{CLASSES[c][:4]}:{logit_diff[c]:+.3f}" for c in range(10)))
    lines.append("")

summary.sort(key=lambda x: x['max_col'])
lines.append("=== Pre-commitment ranking (sorted by max |collision|, CLEAN FL model) ===")
lines.append(f"  {'trigger':12s}  {'max_col':>8s}  {'vs_class':>10s}  {'dominant_pred':>14s}  {'implant':>7s}")
lines.append(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*7}")
for r in summary:
    lines.append(f"  {r['name']:12s}  {r['max_col']:8.4f}  {r['max_cls']:>10s}  "
                 f"{r['dominant']:>14s}  {r['known']:>7s}")

out = '\n'.join(lines)
print(out)
with open(OUT_PATH, 'w') as f:
    f.write(out + '\n')
print(f"\nSaved: {OUT_PATH}")
