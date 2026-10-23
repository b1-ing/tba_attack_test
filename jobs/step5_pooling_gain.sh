#!/bin/bash
#SBATCH --job-name=step5_pool
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Step 2c: pooling gain analysis across trigger variants.

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_pool_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
WARMUP_DIR=$HOME/ffl_check1

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
cp -r $REPO/jobs     $SCRATCH_DIR/

# Use pre-staged CIFAR-10 to avoid downloading
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 data staged"

cd $SCRATCH_DIR

python3 jobs/step5_pooling_gain.py \
    --code-dir   $SCRATCH_DIR/FilterFL \
    --warmup-dir $WARMUP_DIR \
    --out        $SCRATCH_DIR \
    --n-samples  500 \
    --amplitude  0.1145 \
    --seed       1

cp $SCRATCH_DIR/step5_pooling_gain.csv $OUT_DIR/ 2>/dev/null && \
    echo "Copied pooling_gain CSV to $OUT_DIR/"

echo "Done."
