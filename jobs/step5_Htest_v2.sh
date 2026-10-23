#!/bin/bash
#SBATCH --job-name=step5_Htest_v2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

echo "step5_Htest_v2 job $SLURM_JOB_ID on $(hostname)"

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
HTEST_DIR=$HOME/ffl_step5/Htest
FFL_DIR=$REPO/FilterFL
CIFAR10_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py

source $VENV/bin/activate

python3 $REPO/jobs/step5_Htest_analysis_v2.py \
    $HTEST_DIR \
    $FFL_DIR \
    $CIFAR10_DIR \
    2>&1 | tee $HTEST_DIR/step5_Htest_analysis_v2.txt

echo "Done. Output: $HTEST_DIR/step5_Htest_analysis_v2.txt"
