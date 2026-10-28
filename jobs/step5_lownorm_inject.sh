#!/bin/bash
#SBATCH --job-name=step5_lownorm
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Low-norm orthogonal injection test.
# Caps the adversarial client's update norm to TARGET_NORM (×1 benign average, ≈1.2)
# after local training, before submission to the server.
# Tests: can the checker backdoor still implant when orth_norm ≤ 1× benign?
#
# Config: checker, alpha=0.5, ours_standard=[0.5,0.4,0.3,0.2], seed=1,
#         check12b setup, 1200 rounds (more rounds to give quiet injection time).
# target_norm = 1.2 (≈1× observed benign average from suboverlap run)

echo "step5_lownorm job $SLURM_JOB_ID on $(hostname)"

TARGET_NORM=1.2

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_lownorm_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch algorithm.py: cap adversarial update norm before submission ──────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
target_norm = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# Insert norm-cap immediately after w_locals.append(w) for POISON clients.
# We cap ONLY the adversarial clients' update delta to target_norm.
# Find the line that appends to w_locals (the normal non-poison path):
#   w_locals.append(w)
# We need to patch the section that handles both bon_k>1 and the else path.
# Strategy: patch the Aggregation call to cap adv norms on the w_locals list
# before calling Aggregation, using knowledge of adversarial_name_keys.

target = '        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},'

cap_block = f'''\
        # [LOW-NORM CAP] Cap adversarial client update norms to target_norm.
        _lnc_target = {target_norm}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_delta_sq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_delta_norm = _lnc_delta_sq ** 0.5
                if _lnc_delta_norm > _lnc_target:
                    _lnc_scale = _lnc_target / _lnc_delta_norm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_scale
                        ).to(w_locals[_lnc_i][k].dtype)
                    logger.info(f'[LOWNORM] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_delta_norm:.3f}}->'
                                f'{{_lnc_target:.3f}} scale={{_lnc_scale:.4f}}')
        # [END LOW-NORM CAP]
'''

if target not in src:
    print(f'ERROR: patch target not found', flush=True); sys.exit(1)

src = src.replace(target, cap_block + target, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched {alg_path}: adv norm cap at {target_norm} inserted', flush=True)
PATCHEOF

if [[ $? -ne 0 ]]; then echo "Patch failed — aborting"; exit 1; fi

# ── Patch YAML ────────────────────────────────────────────────────────────────
python3 -I - <<PYEOF
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
p['epochs']                       = 1200
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs'] = list(range(50, 1201, 50))
p['probe_save_dir']    = '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/probes_lownorm'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker alpha=0.5 1200 rounds norm_cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_lownorm.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done. Running wfc analysis..."

python3 $REPO/jobs/step5_wfc_analysis.py \
    --probe-dir $HOME/ffl_step5/probes_lownorm \
    --ffl-dir   $REPO/FilterFL \
    --data-dir  $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
    --trigger   checker \
    --out       $OUT_DIR/step5_wfc_lownorm.txt \
    --rounds 50 150 400 600 900 1200 \
    2>&1 | tee $OUT_DIR/step5_wfc_lownorm_analysis.log || true

echo "Done."
echo "  Log:     $LOG"
echo "  Result:  $OUT_DIR/step5_wfc_lownorm.txt"
