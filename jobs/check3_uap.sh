#!/bin/bash --login
#SBATCH --job-name=ffl_check3
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-1:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check3_%j.out

# Check 3: UAP hypothesis test
# Computes a generic UAP toward class 0 on the global warmup model, measures
# each client model's UAP score, and correlates with Stage-3 (GAN probe) score.
#
# Output: $HOME/ffl_check3/check3_uap.{csv,log}
#
# Env overrides:
#   SKIP_WARMUP=1        reuse warmup checkpoint from ffl_check1
#   UAP_EPS=0.1          L-inf UAP budget
#   UAP_STEPS=300        PGD steps
#   RATES="0.3 0.5"      poison rates for adversarial clients
#   ADV_CLIENTS="34 36 45 47"

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
PROBE_NPZ=$HOME/ffl_probe/s1_ours/probes.npz
WARMUP_DIR=$HOME/ffl_check1   # reuse check1's warmup checkpoint

RATES=${RATES:-"0.3 0.5"}
ADV_CLIENTS=${ADV_CLIENTS:-"34 36 45 47"}
UAP_EPS=${UAP_EPS:-0.1}
UAP_STEPS=${UAP_STEPS:-300}

SKIP_ARG=""
if [ "${SKIP_WARMUP:-1}" = "1" ] && [ -f "$WARMUP_DIR/warmup_ckpt.pt" ]; then
    SKIP_ARG="--skip-warmup"
fi

OUT=$HOME/ffl_check3
mkdir -p "$OUT"

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check3_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | rates=$RATES | eps=$UAP_EPS | steps=$UAP_STEPS | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"

python "$PROJ/jobs/check3_uap.py" \
    --code-dir "$WORK" \
    --out "$OUT" \
    --probe-npz "$PROBE_NPZ" \
    --warmup-dir "$WARMUP_DIR" \
    --rates $RATES \
    --adv-clients $ADV_CLIENTS \
    --uap-eps "$UAP_EPS" \
    --uap-steps "$UAP_STEPS" \
    --n-benign 5 \
    $SKIP_ARG \
    2>&1 | tee "$OUT/check3_uap.log"

echo "Done: $(date)"
