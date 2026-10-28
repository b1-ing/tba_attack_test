#!/bin/bash
#SBATCH --job-name=step5_distrib
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Distributional separability analysis.
# Tests whether the low-norm attacker (cap=0.057) is statistically indistinguishable
# from honest non-IID clients on each defense's own detection statistic.
#
# Arms:
#   1: FilterFL — logs per-client s3 scores (adv via FEAT_ALIGN, benign via FEAT_ALIGN_BEN)
#   2: FLAME    — logs per-client norms with CLIENT_FLAGS ordering → norm distribution
#   3: FLTrust  — logs per-client cosine trust with CLIENT_FLAGS ordering → trust distribution
#
# All arms: cap=0.057 (2% benign norm), checker, alpha=0.5, seed=1, 600 rounds.
# Instrumentation: CLIENT_FLAGS log line before each aggregation round gives
#   client ordering + adv_mask, enabling separation of attacker vs honest distributions.
# Post-training: step5_distrib_analysis.py computes AUROC, KS, overlap, threshold sweep.

TARGET_NORM=0.057
DEFENSES=(dummy ours flame fltrust)
DEF_TAGS=(dummy filterfl flame fltrust)
DEFENSE=${DEFENSES[$SLURM_ARRAY_TASK_ID]}
DEF_TAG=${DEF_TAGS[$SLURM_ARRAY_TASK_ID]}

echo "step5_distrib job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "defense=$DEFENSE  cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_distrib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch 1: normcap + CLIENT_FLAGS logging ───────────────────────────────────
# Caps adversarial norm to TARGET_NORM.
# Also logs [CLIENT_FLAGS] ep=... order=[...] adv_mask=[0/1,...] each round,
# enabling post-hoc mapping of per-client defense stats to adv/benign labels.
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

cap_and_flags = f'''\
        # [NORMCAP+FLAGS] Cap adversarial norms; log client ordering for distributional analysis.
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
        # [CLIENT_FLAGS]: per-round client ordering + adv/ben label for distributional analysis.
        _cf_mask = ",".join("1" if n in adversarial_name_keys else "0"
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
print(f'normcap+flags patch applied (target={tn})', flush=True)
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
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['fltrust_server_epochs']        = 2
# Enable benign s3 logging for FilterFL arm only.
p['log_ben_s3']                   = ('$DEFENSE' == 'ours')
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: defense=$DEFENSE  log_ben_s3=' + str('$DEFENSE' == 'ours'), flush=True)
PYEOF

LOG=$OUT_DIR/step5_distrib_${DEF_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense $DEFENSE 2>&1 | tee $LOG

echo ""
echo "Training done."

# ── Distributional analysis ───────────────────────────────────────────────────
OUT_ANALYSIS=$OUT_DIR/step5_distrib_${DEF_TAG}_gate.txt
python3 $REPO/jobs/step5_distrib_analysis.py \
    --defense $DEF_TAG \
    --log     $LOG \
    --out     $OUT_ANALYSIS \
    --cap     $TARGET_NORM \
    2>&1 | tee ${OUT_ANALYSIS%.txt}.log

echo ""
echo "Done."
echo "  Log:      $LOG"
echo "  Analysis: $OUT_ANALYSIS"
