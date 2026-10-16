#!/bin/bash --login
#SBATCH --job-name=ffl_collect
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-00:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_collect_%j.out

# Copy finished probe-run results from $HOME/ffl_probe (+ scratch logs) into the repo at results/probe/<run>/ so they can be
# committed. Per run: probes.npz, params.yaml, log.txt.gz (the reference's own log). Slurm .out files are not copied (log.txt
# has the same content). Nothing is git-added or committed. Safe to re-run; it overwrites the repo copies.

set -uo pipefail
PROJ=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/tba_attack_test
SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
for d in "$HOME"/ffl_probe/*/; do
    name=$(basename "$d")
    [ -f "$d/probes.npz" ] || { echo "SKIP $name (no probes.npz)"; continue; }
    dest=$PROJ/results/probe/$name
    mkdir -p "$dest"
    cp "$d/probes.npz" "$dest/"
    # newest scratch work dir for this run that actually has a reference log (skips crashed attempts)
    log=$(ls -td "$SCRATCH"/ffl_probe_*_"$name"/saved_models/model_*/log.txt 2>/dev/null | head -n1 || true)
    if [ -n "$log" ]; then
        cp "$(dirname "$log")/params.yaml" "$dest/"
        gzip -c "$log" > "$dest/log.txt.gz"
        echo "OK   $name  <- $(dirname "$log")  | $(grep -a count_all_attack "$log" | tail -n1)"
    else
        echo "WARN $name: probes.npz copied but no scratch log found"
    fi
done
du -sh "$PROJ/results/probe"
