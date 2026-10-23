#!/bin/bash
#SBATCH --job-name=step5_Htest
#SBATCH --array=1-3
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# Experiment H: is S3 decay model-driven?
# Three trajectories, 400 rounds FedAvg, per-round state saved at 50/150/250/330/400.
#
# Array:
#   1  checker FedAvg (backdoored trajectory)          → Htest/checker/
#   2  checker FedAvg --benign (clean, no poisoning)   → Htest/clean/
#   3  patch   FedAvg (patch control)                  → Htest/patch/
#
# check12b: 100 clients, 10/round, 10 ADV, poison_rate=0.5,
#   local_epochs=2, lr=0.01, momentum=0.9, seed=1, alpha=0.5.

echo "step5_Htest array $SLURM_ARRAY_TASK_ID on $(hostname)"

ARMS=('' checker clean patch)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_Htest_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

BASE_OUT=$HOME/ffl_step5/Htest
OUT_DIR=$BASE_OUT/$ARM
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"
mkdir -p $SCRATCH_DIR/FilterFL/saved_models

COMMON="--defense fedavg --seed 1 --n-rounds 400 --n-total 100 --n-clients 10 --n-adv 10
        --poison-rate 0.5 --local-epochs 2 --lr 0.01 --momentum 0.9 --alpha 0.5
        --checkpoint-every 50 --save-at-rounds 50,150,250,330,400
        --code-dir $SCRATCH_DIR/FilterFL --out $OUT_DIR"

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    echo "Arm 1: checker FedAvg (backdoored)"
    python3 $REPO/jobs/step5_rerun.py $COMMON --trigger checker

elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    echo "Arm 2: checker FedAvg --benign (clean, no poisoning)"
    python3 $REPO/jobs/step5_rerun.py $COMMON --trigger checker --benign

elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    echo "Arm 3: patch FedAvg (patch control)"
    python3 $REPO/jobs/step5_rerun.py $COMMON --trigger patch
fi

echo "Done. State files in $OUT_DIR:"
ls $OUT_DIR/*state_rnd*.pt 2>/dev/null || echo "(none found)"
