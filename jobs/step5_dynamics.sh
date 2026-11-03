#!/bin/bash
#SBATCH --job-name=step5_dynamics
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Measure backdoor implantation dynamics: m_nat, g_adv.v_m, lambda.
# Compares checker (YES) vs blend_87/blend_75 (NO).
# Output: ~/ffl_step5/step5_dynamics_measure.txt

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

echo "step5_dynamics job $SLURM_JOB_ID on $(hostname)"
$PYTHON /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/jobs/run_dynamics_measure.py
echo "Done."
