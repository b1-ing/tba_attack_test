#!/bin/bash
#SBATCH --job-name=step5_fcfl2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Feature collision analysis only — uses already-saved clean FL model.
# Excludes 'patch' trigger (uses pixel replacement, no additive delta).

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

echo "step5_fcfl2 job $SLURM_JOB_ID on $(hostname)"
$PYTHON /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/jobs/run_featcol_fl.py
echo "Done."
