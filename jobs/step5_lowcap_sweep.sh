#!/bin/bash
#SBATCH --job-name=step5_lowcap
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Low-cap sweep: push norm cap below 0.5x benign to find the implantation floor.
# Benign mean norm ≈ 1.13.
# Arms: 0.1x=0.11  0.2x=0.23  0.3x=0.34
#
# Same setup as step5_normcap_sweep but caps below the current 0.5x floor.
# Config: checker, alpha=0.5, seed=1, 600 rounds, FilterFL defense (ours).

CAPS=(dummy 0.11 0.23 0.34)
TARGET_NORM=${CAPS[$SLURM_ARRAY_TASK_ID]}
CAP_TAG=$(echo "$TARGET_NORM" | tr '.' 'p')

echo "step5_lowcap job $SLURM_JOB_ID array $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "TARGET_NORM=$TARGET_NORM  tag=$CAP_TAG  ($(echo "scale=2; $TARGET_NORM/1.13" | bc)x benign)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_lowcap_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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

# ── Patch: norm-cap ───────────────────────────────────────────────────────────
export SCRATCH_DIR TARGET_NORM
python3 -I - <<'PATCHEOF'
import os, sys

scratch = os.environ['SCRATCH_DIR']
tn      = float(os.environ['TARGET_NORM'])
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

cap_block = f'''\
        # [NORMCAP] Cap adversarial client update norms to target_norm.
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
                    logger.info(f'[NORMCAP] ep={{epoch}} client {{_lnc_name}} '
                                f'norm {{_lnc_delta_norm:.3f}}->'
                                f'{{_lnc_target:.3f}} scale={{_lnc_scale:.4f}}')
        # [END NORMCAP]
'''

insert_marker = '        # ── FLAME / FLTrust aggregation'
if insert_marker not in src:
    print('ERROR: insert marker not found', flush=True); sys.exit(1)

src = src.replace(insert_marker, cap_block + insert_marker, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched {alg_path}: norm cap at {tn}', flush=True)
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
p['probe_save_dir']               = '$OUT_DIR/probes_lowcap_$CAP_TAG'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker 600r cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_lowcap_${CAP_TAG}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done."

# ── Summary ───────────────────────────────────────────────────────────────────
python3 -I - <<SUMEOF
import re

log_path = '$LOG'
with open(log_path) as f:
    lines = f.readlines()

asr_lines   = [l for l in lines if 'Global_Test_Poison:True'  in l]
clean_lines = [l for l in lines if 'Global_Test_Poison:False' in l]
def_lines   = [l for l in lines if 'count_all_attack' in l]

asr_final   = asr_lines[-1].strip()   if asr_lines   else 'N/A'
clean_final = clean_lines[-1].strip() if clean_lines  else 'N/A'

tot_atk = tot_def = cor_def = 0
if def_lines:
    m = re.search(r'count_all_attack: (\d+), count_all_defense: (\d+), count_correct_defense: (\d+)', def_lines[-1])
    if m:
        tot_atk, tot_def, cor_def = int(m.group(1)), int(m.group(2)), int(m.group(3))
det_rate = cor_def / tot_atk if tot_atk > 0 else float('nan')

# ASR trajectory: every 50th round
asr_traj = [(i+1, l.strip()) for i, l in enumerate(asr_lines) if (i+1) % 50 == 0]

out = f'''=== lowcap arm: TARGET_NORM=$TARGET_NORM ({TARGET_NORM:.2f} = {float("$TARGET_NORM")/1.13:.2f}x benign) ===
Final ASR:       {asr_final}
Final clean:     {clean_final}
Det. rate:       {cor_def}/{tot_atk} = {det_rate:.3f}
ASR trajectory (every 50 rounds):
''' + '\n'.join(f'  r{r*10}: {l}' for r, l in asr_traj)

print(out)
with open('$OUT_DIR/step5_lowcap_${CAP_TAG}_summary.txt', 'w') as f:
    f.write(out)
SUMEOF

echo "Done."
echo "  Log:     $LOG"
echo "  Summary: $OUT_DIR/step5_lowcap_${CAP_TAG}_summary.txt"
