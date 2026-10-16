#!/bin/bash --login
#SBATCH --job-name=ffl_check1
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-2:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_check1_%j.out

# Check 1 (corrected): measure prior / local-ASR / Stage-3 per poison rate
# from a common warm-up checkpoint. Also measures benign client prior (Check 2 ext).
#
# Output: $HOME/ffl_check1/check1_prior.{csv,png,log}
#
# Optional env overrides (--export=ALL,...):
#   WARMUP=N          number of clean FL warm-up rounds (default 50)
#   RATES="0.1 0.25 0.3 0.5"
#   SKIP_WARMUP=1     reuse existing warmup_ckpt.pt in output dir
#   N_BENIGN=5        how many top-flagged benign clients to evaluate

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
DATA_SRC=$HOME/filterfl_test/filterfl_test/data
PROBE_NPZ=$HOME/ffl_probe/s1_ours/probes.npz

WARMUP=${WARMUP:-50}
RATES=${RATES:-"0.1 0.2 0.25 0.3 0.4 0.5"}
N_BENIGN=${N_BENIGN:-5}
SKIP_ARG=""
if [ "${SKIP_WARMUP:-0}" = "1" ]; then SKIP_ARG="--skip-warmup"; fi

OUT=$HOME/ffl_check1
mkdir -p "$OUT"

if [ ! -f "$PROJ/FilterFL/main.py" ]; then
    echo "ERROR: $PROJ/FilterFL/main.py missing" >&2; exit 1
fi

module purge
module load miniforge/24.11.2
source "$VENV/bin/activate"

WORK=$SCRATCH/ffl_check1_${SLURM_JOB_ID}
mkdir -p "$WORK"
cp -r "$PROJ/FilterFL/." "$WORK/"
cd "$WORK"
rm -rf data && mkdir -p data
cp -r "$DATA_SRC/cifar-10-batches-py" "$DATA_SRC/cifar-10-python.tar.gz" data/

echo "Host: $(hostname) | warmup=$WARMUP rates=$RATES | start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"

python "$PROJ/jobs/check1_prior.py" \
    --code-dir "$WORK" \
    --out "$OUT" \
    --probe-npz "$PROBE_NPZ" \
    --warmup-rounds "$WARMUP" \
    --rates $RATES \
    --n-benign "$N_BENIGN" \
    $SKIP_ARG \
    2>&1 | tee "$OUT/check1_prior.log"

echo "Done: $(date)"
