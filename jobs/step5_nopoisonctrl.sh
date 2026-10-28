#!/bin/bash
#SBATCH --job-name=step5_noctrl
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# No-poison control: pure benign FedAvg (is_poison=False), 600 rounds.
# After training, evaluate checker-triggered ASR and clean target-class rate
# on the final global model.
#
# Purpose: if checker-ASR is high here, it is STRUCTURAL (benign training on
# CIFAR-10 with this architecture produces a model that classifies triggered
# non-target images as target class 0 regardless of any attack). That would
# make the sweep ASRs uninterpretable.
#
# Also records the clean target-class rate (fraction of non-target test images
# classified as target WITHOUT the trigger) as a baseline false-positive rate.

echo "step5_noctrl job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_noctrl_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── YAML: benign FedAvg, no poisoning ─────────────────────────────────────────
python3 -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10   # doesn't matter; is_poison=False
p['is_poison']                    = False
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = 600
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: is_poison=False 600r (no-poison control)', flush=True)
PYEOF

LOG=$OUT_DIR/step5_noctrl.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done. Running post-hoc ASR + clean-target-rate eval..."

# Save final global model for eval
CKPT=$OUT_DIR/step5_noctrl_final.pt
python3 -I - <<SAVEEOF
import glob, os, torch

# main.py saves checkpoints under saved_models/
ckpts = sorted(glob.glob('saved_models/*.pt'))
if ckpts:
    import shutil
    shutil.copy(ckpts[-1], '$CKPT')
    print(f'Copied {ckpts[-1]} -> $CKPT')
else:
    print('No checkpoint found in saved_models/')
SAVEEOF

# Post-hoc eval: checker-ASR and clean-target-rate on final model
python3 -I - <<EVALEOF
import pickle, numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, '$SCRATCH_DIR/FilterFL')
from models.ResNet8 import ResNet8
from trigger_utils import make_raw_delta, apply_additive_norm

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET = 0
AMP_RAW = 0.1145

ckpt = '$CKPT'
data_dir = '$DATA_DIR'

try:
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
except Exception as e:
    print(f'Could not load checkpoint: {e}')
    sys.exit(0)

m = ResNet8(num_classes=10)
m.load_state_dict(sd)
m.eval()

# Load CIFAR-10 test set
with open(f'{data_dir}/test_batch', 'rb') as f:
    d = pickle.load(f, encoding='bytes')
imgs_raw = d[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
labels   = np.array(d[b'labels'])

# Non-target images only
nontarget_idx = np.where(labels != TARGET)[0]
imgs_nt = imgs_raw[nontarget_idx]   # (N, 3, 32, 32), raw [0,1]
labels_nt = labels[nontarget_idx]

def norm(x):
    return (x - MEAN[:, None, None]) / STD[:, None, None]

def to_tensor(x_batch):
    return torch.tensor(np.stack([norm(img) for img in x_batch]), dtype=torch.float32)

# Apply checker trigger
delta = make_raw_delta('checker', AMP_RAW, seed=1)
def apply_trigger(imgs):
    out = []
    for img in imgs:
        t = np.clip(img + delta, 0, 1)
        out.append(t)
    return np.array(out)

imgs_triggered = apply_trigger(imgs_nt)

BS = 200
def predict(imgs):
    preds = []
    with torch.no_grad():
        for i in range(0, len(imgs), BS):
            x = to_tensor(imgs[i:i+BS])
            logits = m(x)
            preds.extend(logits.argmax(1).tolist())
    return np.array(preds)

preds_clean    = predict(imgs_nt)
preds_trigger  = predict(imgs_triggered)

clean_target_rate = (preds_clean == TARGET).mean()
asr_raw           = (preds_trigger == TARGET).mean()
net_asr           = asr_raw - clean_target_rate

print(f'=== NO-POISON CONTROL (final model, ep=600) ===')
print(f'N non-target test images: {len(imgs_nt)}')
print(f'Clean target-class rate (no trigger): {clean_target_rate:.4f}  ({(preds_clean==TARGET).sum()}/{len(imgs_nt)})')
print(f'Triggered ASR (raw):                  {asr_raw:.4f}  ({(preds_trigger==TARGET).sum()}/{len(imgs_nt)})')
print(f'Net ASR (triggered - clean):          {net_asr:.4f}')
print()
if asr_raw > 0.5:
    print('WARNING: high ASR in no-poison control — may be structural')
else:
    print('OK: low ASR in no-poison control — sweep ASRs are attack-driven')

with open('$OUT_DIR/step5_noctrl_eval.txt', 'w') as f:
    f.write(f'clean_target_rate={clean_target_rate:.6f}\n')
    f.write(f'asr_raw={asr_raw:.6f}\n')
    f.write(f'net_asr={net_asr:.6f}\n')
    f.write(f'n_nontarget={len(imgs_nt)}\n')
EVALEOF

echo "Done."
echo "  Log:  $LOG"
echo "  Eval: $OUT_DIR/step5_noctrl_eval.txt"
