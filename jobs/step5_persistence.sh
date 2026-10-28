#!/bin/bash
#SBATCH --job-name=step5_persist
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Quiet-injection persistence test.
# Phase 1 (r1-150):  attacker active, norm cap at 1.2 (1× benign)
# Phase 2 (r151-600): attacker DISABLED - all clients train benign
# Measures ASR decay after attack stops to test persistence vs continuous-injection artifact.
# Compares to loud checker half-life of 379 rounds.
#
# Config: checker, alpha=0.5, ours_standard=[0.5,0.4,0.3,0.2], seed=1, 600 rounds.

echo "step5_persist job $SLURM_JOB_ID on $(hostname)"

STOP_ATTACK_EP=150
TARGET_NORM=1.2

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_persist_${SLURM_JOB_ID}
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

# ── Patch algorithm.py: norm cap + attack stop ─────────────────────────────────
export SCRATCH_DIR STOP_ATTACK_EP TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch      = os.environ['SCRATCH_DIR']
stop_ep      = int(os.environ['STOP_ATTACK_EP'])
target_norm  = float(os.environ['TARGET_NORM'])
alg_path     = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

target = '        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},'

patch = f'''\
        # [PERSIST] Phase switch: disable attack after stop_attack_epoch.
        _persist_stop = {stop_ep}
        if epoch > _persist_stop:
            adversarial_name_keys = set()   # treat all clients as benign this round

        # [LOWNORM CAP] Cap adv update norms to target_norm before submission.
        _lnc_target = {target_norm}
        _lnc_keys = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _lnc_dq = sum(
                    (w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                    for k in _lnc_keys)
                _lnc_dn = _lnc_dq ** 0.5
                if _lnc_dn > _lnc_target:
                    _lnc_sc = _lnc_target / _lnc_dn
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] +
                            (w_locals[_lnc_i][k] - ori_weight[k]) * _lnc_sc
                        ).to(w_locals[_lnc_i][k].dtype)
                    logger.info(f'[LOWNORM] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_dn:.3f}}->'
                                f'{{_lnc_target:.3f}}')
        # [END PERSIST/LOWNORM]
'''

if target not in src:
    print('ERROR: patch target not found', flush=True); sys.exit(1)

src = src.replace(target, patch + target, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched: attack stops ep>{stop_ep}, norm cap={target_norm}', flush=True)
PATCHEOF

if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

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
p['epochs']                       = 600
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = list(range(50, 601, 50))
p['probe_save_dir']               = '$OUT_DIR/probes_persist'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker alpha=0.5 600r attack_stop=150 norm_cap=1.2', flush=True)
PYEOF

mkdir -p $OUT_DIR/probes_persist
LOG=$OUT_DIR/step5_persist.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo "Done."
echo "  Log: $LOG"
