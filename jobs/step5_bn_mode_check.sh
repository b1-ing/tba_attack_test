#!/bin/bash
#SBATCH --job-name=step5_bn_mode
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

echo "step5_bn_mode job $SLURM_JOB_ID on $(hostname)"

source ~/venvs/tba_attack/bin/activate

python3 ~/tba_attack_test/jobs/step5_bn_mode_check.py \
    ~/ffl_step5/probes_checker2 \
    ~/tba_attack_test/FilterFL \
    2>&1 | tee ~/ffl_step5/step5_bn_mode_check.txt

echo "Done: ~/ffl_step5/step5_bn_mode_check.txt"
