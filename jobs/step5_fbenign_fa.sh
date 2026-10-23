#!/bin/bash
#SBATCH --job-name=step5_fbenign_fa
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# F-benign: FEAT_ALIGN vs s3 (train mode) for benign + adversarial clients.
#
# Clients:
#   adversarial:     34, 36, 45, 47 at rates 0.3, 0.5
#   benign_skewed:   73, 35, 89, 49, 92, 31, 55, 59 (high class-0 share)
#   benign_balanced: 1, 29, 66, 90 (class-0 share = 0.0)
#
# Global state: warmup_ckpt.pt from check1 (50-round FedAvg warmup, no poison).
# BN mode: TRAIN (matches FilterFL Stage-3). Eval-mode also logged for check11 audit.
# GAN probe (FilterFL default).
# check12b setup: 100 clients, local_epochs=2, lr=0.01, momentum=0.9, seed=1.

echo "step5_fbenign_fa job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_fbenign_fa_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_check_fbenign
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP=$HOME/ffl_check1/warmup_ckpt.pt

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"
mkdir -p $SCRATCH_DIR/FilterFL/saved_models

LOG=$OUT_DIR/step5_fbenign_fa.log

cd $SCRATCH_DIR/FilterFL
python3 $REPO/jobs/step5_fbenign_fa.py \
    --code-dir $SCRATCH_DIR/FilterFL \
    --warmup   $WARMUP \
    --out      $OUT_DIR \
    2>&1 | tee $LOG

echo "Done."
echo "  CSV:  $OUT_DIR/fbenign_fa.csv"
echo "  Log:  $LOG"
