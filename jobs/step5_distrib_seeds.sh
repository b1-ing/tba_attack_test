#!/bin/bash
#SBATCH --job-name=step5_dseed
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-6
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Seed verification: repeat distributional separability analysis at seeds 2 and 3.
# Goal: confirm the FP cost spectrum (FLAME/FilterFL/FLTrust) is stable across seeds.
# Key claims to verify:
#   - FilterFL: ~9% honest exclusion at 80% detection (costly, not impossible)
#   - FLTrust: ~60% honest exclusion at 50% detection (genuinely no usable threshold)
#   - FLAME: 0% honest exclusion with floor fix (patchable)
#
# Array:
#   1: FilterFL seed=2   4: FilterFL seed=3
#   2: FLAME     seed=2   5: FLAME     seed=3
#   3: FLTrust   seed=2   6: FLTrust   seed=3
#
# Config: cap=0.057, checker, alpha=0.5, 600 rounds.

SEEDS=(dummy 2 2 2 3 3 3)
DEFENSES=(dummy ours flame fltrust ours flame fltrust)
DEF_TAGS=(dummy filterfl flame fltrust filterfl flame fltrust)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}
TARGET_NORM=0.057

echo "step5_dseed job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE seed=$SEED cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_dseed_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch: normcap + CLIENT_FLAGS logging ─────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys
scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'
with open(alg_path) as f:
    src = f.read()

cap_and_flags = f'''\
        # [NORMCAP+FLAGS] Cap adversarial norms; log client ordering.
        _lnc_target = {tn}
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
        _cf_mask  = ",".join("1" if n in adversarial_name_keys else "0"
                             for n in agent_name_keys)
        _cf_order = ",".join(str(n) for n in agent_name_keys)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} order=[{{_cf_order}}] adv_mask=[{{_cf_mask}}]')
        # [END NORMCAP+FLAGS]
'''
marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found', flush=True); sys.exit(1)
src = src.replace(marker, cap_and_flags + marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'patch applied (target={tn})', flush=True)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ──────────────────────────────────────────────────────────────────────
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
p['seed']                         = $SEED
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
p['log_ben_s3']                   = ('$DEFENSE' == 'ours')
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok seed=$SEED defense=$DEFENSE', flush=True)
PYEOF

LOG=$OUT_DIR/step5_distrib_${DEF_TAG}_s${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo "Training done."

# ── Distributional analysis ───────────────────────────────────────────────────
OUT_ANALYSIS=$OUT_DIR/step5_distrib_${DEF_TAG}_s${SEED}_gate.txt
python3 $REPO/jobs/step5_distrib_analysis.py \
    --defense $DEF_TAG \
    --log     $LOG \
    --out     $OUT_ANALYSIS \
    --cap     $TARGET_NORM \
    2>&1 | tee ${OUT_ANALYSIS%.txt}.log

echo "Done."
echo "  Log:      $LOG"
echo "  Analysis: $OUT_ANALYSIS"
