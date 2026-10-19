#!/bin/bash --login
#SBATCH --job-name=ffl_check4
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-5:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check4_%j.out

# Check 4: single-client UAP-suppression sweep.
# Trains adversarial client 45 at rate=0.5 with uap_mu in [0.0, 0.01, 0.1, 0.5].
# Reports UAP score, Stage-3 score, and local ASR for each mu.
# Use to decide whether the suppression objective is compatible with the backdoor
# before committing to a full probe run.
#
# Env overrides:
#   ADV_CLIENT=45      which adversarial client to test
#   RATE=0.5           poison rate
#   MU_VALUES="0.0 0.01 0.1 0.5"

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
WARMUP_DIR=$HOME/ffl_check1   # reuse check1 warmup checkpoint

ADV_CLIENTS=${ADV_CLIENTS:-"34 36 45 47"}
RATE=${RATE:-0.5}
MU_VALUES=${MU_VALUES:-"0.0 0.01 0.1 0.5"}
LOCAL_SEEDS=${LOCAL_SEEDS:-"1 2 3"}

OUT=$HOME/ffl_check4
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check4_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | clients=$ADV_CLIENTS | rate=$RATE | mu_values=$MU_VALUES | seeds=$LOCAL_SEEDS | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"

python "$PROJ/jobs/check4_uap_suppress.py" \
    --code-dir "$WORK" \
    --out "$OUT" \
    --warmup-dir "$WARMUP_DIR" \
    --skip-warmup \
    --adv-clients $ADV_CLIENTS \
    --rate "$RATE" \
    --mu-values $MU_VALUES \
    --local-seeds $LOCAL_SEEDS \
    --uap-eps 0.1 \
    --uap-steps 300 \
    --uap-steps-train 30 \
    2>&1 | tee "$OUT/check4_uap_suppress.log"

echo "Done: $(date)"
