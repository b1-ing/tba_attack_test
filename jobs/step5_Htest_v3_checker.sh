#!/bin/bash
#SBATCH --job-name=step5_Htest_v3_chk
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

echo "step5_Htest_v3_checker job $SLURM_JOB_ID on $(hostname)"

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

python3 $REPO/jobs/step5_Htest_analysis_v3.py \
    $HOME/ffl_step5/probes_checker \
    $REPO/FilterFL \
    checker \
    2>&1 | tee $HOME/ffl_step5/step5_Htest_analysis_v3_checker.txt

echo "Done. Output: $HOME/ffl_step5/step5_Htest_analysis_v3_checker.txt"
