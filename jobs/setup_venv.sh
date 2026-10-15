#!/bin/bash --login
#SBATCH --job-name=ffl_ref_venv
#SBATCH -p serial
#SBATCH -t 0-1:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_ref_venv_%j.out

# Build a clean venv for running the unmodified reference FilterFL
# (reference/filterfl). CPU-only job: the cu121 wheels bundle their own CUDA
# runtime, so no GPU node or cuda module is needed to install them.
# The venv lives in $HOME (scratch is auto-cleaned). Safe to re-run: it deletes
# and recreates the venv.

set -euo pipefail

PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
VENV=$HOME/venvs/tba_attack

module purge
module load miniforge/24.11.2

PY=$(command -v python3)
echo "Base python: $PY ($($PY --version))"

rm -rf "$VENV"
mkdir -p "$(dirname "$VENV")"
$PY -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip
# Same pins as the main project (torch 2.5.1+cu.
# numpy is imported by the reference code but n.
python -m pip install -r "$PROJ/requirements.txt" numpy

echo "--- installed ---"
python -m pip list 2>/dev/null | grep -i -E "^(torch|torchvision|pyyaml|numpy) "

echo "--- import check (no GPU on this node, so cuda = False is expected)"
python - <<'EOF'
import sys, torch, torchvision, yaml, numpy
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("torchvision", torchvision.__version__, "numpy", numpy.__version, "| pyyaml", yaml.__version__)
EOF

echo "Done: $(date)"
echo "Activate in job scripts with:"
echo "  module purge; module load miniforge/24.11.2/source $VENV/bin/activate"
