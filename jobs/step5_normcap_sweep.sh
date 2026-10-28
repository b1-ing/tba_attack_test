#!/bin/bash
#SBATCH --job-name=step5_normcap
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Norm-cap sweep: vary attacker update norm cap across fractions of the benign mean norm.
# Benign mean delta norm ≈ 1.13 (measured from step5_suboverlap run).
# Arms: 0.5x=0.57  0.8x=0.90  1.0x=1.13  1.2x=1.36  1.5x=1.70
#
# Purpose: find the implantation floor (lowest cap with high final ASR) and the detection
# floor (cap at which stage-3 detection rate rises). The gap between them = the quiet regime.
#
# Config: checker, alpha=0.5, seed=1, 600 rounds, FilterFL defense (ours).

CAPS=(dummy 0.57 0.90 1.13 1.36 1.70)
TARGET_NORM=${CAPS[$SLURM_ARRAY_TASK_ID]}
CAP_TAG=$(echo "$TARGET_NORM" | tr '.' 'p')   # e.g. 1p13

echo "step5_normcap job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "TARGET_NORM=$TARGET_NORM  tag=$CAP_TAG"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_normcap_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: norm-cap on adversarial client deltas ──────────────────────────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
target_norm = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

target = '        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},'

cap_block = f'''\
        # [NORMCAP] Cap adversarial client update norms to target_norm before aggregation.
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
                    logger.info(f'[NORMCAP] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_delta_norm:.3f}}->'
                                f'{{_lnc_target:.3f}} scale={{_lnc_scale:.4f}}')
        # [END NORMCAP]
'''

# Insert before the first Aggregation call (inside the aggregation block, not the ours block)
insert_marker = '        # ── FLAME / FLTrust aggregation'
if insert_marker not in src:
    print('ERROR: insert marker not found', flush=True); sys.exit(1)

src = src.replace(insert_marker, cap_block + insert_marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched {alg_path}: norm cap at {target_norm} inserted', flush=True)
PATCHEOF

if [[ $? -ne 0 ]]; then echo "Patch failed — aborting"; exit 1; fi

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
p['probe_save_epochs']            = [50, 150, 300, 600]
p['probe_save_dir']               = '$OUT_DIR/probes_normcap_$CAP_TAG'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker alpha=0.5 600r norm_cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_normcap_${CAP_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done."

# ── Summary: extract final ASR + detection rate ───────────────────────────────
python3 -I - <<SUMEOF
import re, sys

log_path = '$LOG'
with open(log_path) as f:
    lines = f.readlines()

# Last Global_Test_Poison line = final ASR
asr_lines   = [l for l in lines if 'Global_Test_Poison:True' in l]
clean_lines = [l for l in lines if 'Global_Test_Poison:False' in l]
def_lines   = [l for l in lines if 'count_all_attack' in l]

asr_final   = asr_lines[-1].strip()   if asr_lines   else 'N/A'
clean_final = clean_lines[-1].strip() if clean_lines  else 'N/A'
def_final   = def_lines[-1].strip()   if def_lines    else 'N/A'

# Parse detection rate
tot_atk = tot_def = cor_def = 0
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)', def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))

det_rate = cor_def / tot_atk if tot_atk > 0 else float('nan')

out = f'''=== normcap sweep arm: TARGET_NORM=$TARGET_NORM ({TARGET_NORM:.2f} = {TARGET_NORM/1.13:.2f}x benign mean) ===
Final ASR:        {asr_final}
Final clean:      {clean_final}
Defense counts:   {def_final}
Detection rate:   {cor_def}/{tot_atk} = {det_rate:.3f}
'''
print(out)
with open('$OUT_DIR/step5_normcap_${CAP_TAG}_summary.txt', 'w') as f:
    f.write(out)
SUMEOF

echo "Done."
echo "  Log:     $LOG"
echo "  Summary: $OUT_DIR/step5_normcap_${CAP_TAG}_summary.txt"
