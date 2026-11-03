#!/bin/bash
#SBATCH --job-name=step5_patch
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Held-out test of the inference-margin / max_col criterion for the patch trigger.
# Step 1: Compute max_col and feat_norm of patch on clean ResNet8 (one forward pass).
#   Patch = 16 pixel-replacement positions (rows 0,3 x cols 0-3,6-9), set to 1.0 in norm space.
#   Uses step5_clean_resnet8.pt (same reference model as margin probe).
# Step 2: 600-round normcap retest (cap=0.057, defense=ours, seed=1, fixed test.py).
#   Checks if 1.8% ASR holds under the fixed evaluation.
# Prediction: feat_norm healthy (~7-9) + high max_col → collision failure (same as stripes).

TRIGGER=patch
SEED=1
TARGET_NORM=0.057
ROUNDS=600

echo "step5_patch job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_patch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
CLEAN_MODEL=$OUT_DIR/step5_clean_resnet8.pt
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: python not found"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Step 1: Compute max_col and feat_norm for patch trigger ────────────────────
echo "=== Computing patch max_col and feat_norm ==="
MAXCOL_OUT=$OUT_DIR/step5_patch_maxcol.txt
$PYTHON -W ignore - <<MCEOF
import sys, os, pickle
sys.path.insert(0, '$SCRATCH_DIR/FilterFL')
import torch, torch.nn.functional as F, numpy as np

TARGET = 0  # airplane
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

# Patch positions from cifar_params.yaml:
# row 0: cols 0,1,2,3,6,7,8,9  |  row 3: same
PATCH_POS = [(0,0),(0,1),(0,2),(0,3),(0,6),(0,7),(0,8),(0,9),
             (3,0),(3,1),(3,2),(3,3),(3,6),(3,7),(3,8),(3,9)]

# Load clean model
from models.ResNet8 import ResNet8
model = ResNet8(num_classes=10)
sd = torch.load('$CLEAN_MODEL', map_location='cpu', weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()

# Move to GPU if available
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(dev)

# Feature extractor (before final linear)
def get_features(m, x):
    out = F.relu(m.bn1(m.conv1(x)))
    out = m.layer1(out)
    out = m.layer2(out)
    out = m.layer3(out)
    out = m.avgpool(out)
    return torch.flatten(out, 1)  # (N, 512)

# Load test images (non-target)
with open('$DATA_DIR/test_batch', 'rb') as f:
    td = pickle.load(f, encoding='bytes')
imgs_raw = td[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels   = np.array(td[b'labels'])
nontgt   = imgs_raw[labels != TARGET][:500]

# Normalize
mn_t = torch.tensor(MEAN, dtype=torch.float32).to(dev)
st_t = torch.tensor(STD,  dtype=torch.float32).to(dev)
imgs_n = ((torch.tensor(nontgt) - torch.tensor(MEAN)) / torch.tensor(STD)).float().to(dev)

# Apply patch: set specified positions to 1.0 in normalized space (matches test.py behaviour)
imgs_p = imgs_n.clone()
for (r, c) in PATCH_POS:
    imgs_p[:, :, r, c] = 1.0  # all channels

with torch.no_grad():
    feat_clean   = get_features(model, imgs_n)   # (N, 512)
    feat_patched = get_features(model, imgs_p)   # (N, 512)

feat_shift = (feat_patched - feat_clean)  # (N, 512)
mean_shift = feat_shift.mean(dim=0).cpu().numpy()  # (512,)
feat_norm  = float(np.linalg.norm(mean_shift))

# FC weight matrix
W = sd['linear1.weight'].numpy()          # (10, 512)
W_norm = W / np.linalg.norm(W, axis=1, keepdims=True)

if feat_norm > 1e-9:
    d_unit = mean_shift / feat_norm
    cos_per_class = np.abs(W_norm @ d_unit)   # (10,)
    max_col   = float(cos_per_class.max())
    best_class = int(cos_per_class.argmax())
else:
    cos_per_class = np.zeros(10)
    max_col = 0.0
    best_class = -1

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

lines = [
    '=== max_col probe: patch trigger ===',
    f'  feat_norm (mean feature shift L2): {feat_norm:.4f}',
    f'  max_col: {max_col:.4f}',
    f'  best_class: {best_class} ({CIFAR10_CLASSES[best_class] if best_class >= 0 else "n/a"})',
    f'  cos per class: {[round(float(x),3) for x in cos_per_class]}',
    '',
    '  Interpretation:',
]
if feat_norm < 1.0:
    lines.append('  WEAK-SIGNAL: feat_norm < 1.0 — patch produces tiny feature shift.')
    lines.append('  max_col unreliable for this trigger. Failure mode unclear.')
elif feat_norm < 3.0:
    lines.append('  MARGINAL feat_norm — max_col may be unreliable.')
else:
    if max_col > 0.35:
        lines.append('  HEALTHY feat_norm + HIGH max_col → collision failure predicted.')
    else:
        lines.append('  HEALTHY feat_norm + LOW max_col → normcap floor / other mechanism.')

out = '\n'.join(lines)
print(out)
with open('$MAXCOL_OUT', 'w') as f:
    f.write(out + '\n')
print(f'Saved: $MAXCOL_OUT')
MCEOF

cat $MAXCOL_OUT
echo ""

# ── Step 2: Patch normcap + normcap patch ──────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP]
        import numpy as _np_nc
        _lnc_target = {tn}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_norm = _lnc_sq ** 0.5
                if _lnc_norm > _lnc_target:
                    _sc = _lnc_target / _lnc_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] + (w_locals[_lnc_i][k] - ori_weight[k]) * _sc
                        ).to(w_locals[_lnc_i][k].dtype)
        # [END NORMCAP]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
try:
    py_compile.compile(alg_path, doraise=True)
    print(f'normcap patch applied (cap={tn}): compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ─────────────────────────────────────────────────────────────────────
$PYTHON -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = $ROUNDS
p['seed']                         = $SEED
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['trigger_type']                 = 'patch'
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
# keep default 0_poison_pattern (16-pixel positions from yaml)
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok trigger=patch seed=$SEED cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_patch_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense ours 2>&1 | tee $LOG

echo "Training done."

# ── Summary ───────────────────────────────────────────────────────────────────
$PYTHON -I -W ignore - <<SUMEOF
import re, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

CTRL    = 0.019
LOG_PATH = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_patch_seed1.log'
OUT_FILE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_patch_seed1_summary.txt'
MAXCOL_FILE = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_patch_maxcol.txt'

with open(LOG_PATH) as f:
    LOG = f.read()

evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs, i = [], 0
while i + 1 < len(evals):
    if evals[i][0] == 'False' and evals[i+1][0] == 'True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

stable = None
for j, (c, a) in enumerate(pairs):
    if a - CTRL > 0.50:
        future = pairs[j+1:j+11]
        if all(fa - CTRL > 0.30 for _, fa in future):
            stable = j + 1; break

result = 'YES' if stable else 'NO'
fc, fa = pairs[-1] if pairs else (0, 0)
f100_net = np.mean([max(0, a - CTRL) for _, a in pairs[-100:]]) if len(pairs) >= 100 else float('nan')

# Read maxcol result
mc_lines = open(MAXCOL_FILE).read() if __import__('os').path.exists(MAXCOL_FILE) else '(no maxcol file)'

lines = [
    '=== PATCH normcap retest: seed=1 cap=0.057 (FIXED test.py) ===',
    '',
    mc_lines.strip(),
    '',
    f'  RESULT: {result}',
    f'  Final:  clean={fc:.3f}  asr={fa:.3f}  net_asr={max(0, fa-CTRL):.3f}',
    f'  F100 mean net_asr: {f100_net:.3f}',
    f'  Stable impl eval: {stable}',
    '',
    '  ASR trajectory (every 100 evals):',
]
for ep in list(range(0, len(pairs), 100)) + ([len(pairs)-1] if pairs else []):
    c2, a2 = pairs[ep]
    lines.append(f'    eval={ep+1:4d}  clean={c2:.3f}  asr={a2:.3f}  net_asr={max(0,a2-CTRL):.3f}')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
