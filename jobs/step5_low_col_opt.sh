#!/bin/bash
#SBATCH --job-name=step5_lco_opt
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Optimise trigger delta to minimise max feature-collision score.
# Saves low_col_opt_delta.npy to ~/ffl_step5/.
# Submit the FL job with --dependency=afterok:<this_job_id>.

PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12
export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

echo "step5_lco_opt job $SLURM_JOB_ID on $(hostname)"
$PYTHON /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test/jobs/run_low_col_opt.py
echo "Done."
