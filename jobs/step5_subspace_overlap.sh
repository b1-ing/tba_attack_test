#!/bin/bash
#SBATCH --job-name=step5_suboverlap
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --output=%x_%j.out
#SBATCH --exclude=node871,node877

# Subspace overlap measurement: does the adversarial weight update lie in the
# span of benign client updates?
#
# Patches algorithm.py to save all client deltas (adv + benign) at probe_save_epochs,
# then runs step5_subspace_overlap.py to compute per-round overlap fraction.
#
# Setup: checker, alpha=0.5, ours_standard=[0.5,0.4,0.3,0.2], seed=1,
#        check12b (100 clients, 10/round, 10 ADV, poison_rate=0.5), 600 rounds.

echo "step5_suboverlap job $SLURM_JOB_ID on $(hostname)"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_suboverlap_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
PROBE_DIR=$OUT_DIR/probes_suboverlap
mkdir -p $OUT_DIR $PROBE_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack

source $VENV/bin/activate

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py \
      $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch algorithm.py: add client-delta saving before Aggregation call ──────
export SCRATCH_DIR
python3 -I - <<'PATCHEOF'
import re

path = '/tmp/step5_suboverlap_alg_target'
import os, sys
scratch = os.environ['SCRATCH_DIR']
alg_path = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

# Insert delta-saving block immediately before the Aggregation call.
# Match the line that starts the Aggregation call.
target = '        w_glob = Aggregation({k: v.clone() for k, v in ori_weight.items()},'

insert = '''\
        # [SUBSPACE OVERLAP] Save per-client weight deltas at probe_save_epochs.
        _so_epset = set(handle.params.get('probe_save_epochs', []))
        if epoch in _so_epset:
            import os as _sos
            _sdir = handle.params.get('probe_save_dir', '.')
            _sos.makedirs(_sdir, exist_ok=True)
            _adv_deltas, _ben_deltas = [], []
            for _ci, _cclient in enumerate(agent_name_keys):
                _dw = {_k: (w_locals[_ci][_k] - ori_weight[_k]).cpu().float()
                       for _k in ori_weight if ori_weight[_k].dtype != torch.int64}
                if _cclient in adversarial_name_keys:
                    _adv_deltas.append(_dw)
                else:
                    _ben_deltas.append(_dw)
            _dpath = _sos.path.join(_sdir, f'client_deltas_ep{epoch:04d}.pt')
            torch.save({'adv': _adv_deltas, 'ben': _ben_deltas,
                        'epoch': epoch, 'n_adv': len(_adv_deltas)}, _dpath)
            logger.info(f'CLIENT_DELTAS_SAVED ep={epoch} n_adv={len(_adv_deltas)} '
                        f'n_ben={len(_ben_deltas)} path={_dpath}')
        # [END SUBSPACE OVERLAP]
'''

if target not in src:
    print(f'ERROR: patch target not found in {alg_path}', flush=True)
    sys.exit(1)

src = src.replace(target, insert + target, 1)
with open(alg_path, 'w') as f:
    f.write(src)
print(f'Patched {alg_path}: client-delta saving inserted before Aggregation', flush=True)
PATCHEOF

if [[ $? -ne 0 ]]; then
    echo "Patch failed — aborting"
    exit 1
fi

# ── Patch YAML ────────────────────────────────────────────────────────────────
python3 -I - <<PYEOF
import yaml
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
p['number_of_total_participants'] = 100
p['num_clients']                  = 10
p['poison_client_num']            = 10
p['local_epochs']                 = 2
p['lr']                           = 0.01
p['momentum']                     = 0.9
p['poison_rate']                  = 0.5
p['epochs']                       = 600
p['seed']                         = 1
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['probe_mode']                   = 'pixel'
p['log_ben_s3']                   = True
p['trigger_type']                 = 'checker'
p['trigger_amplitude_raw']        = 0.1145
p['probe_save_epochs']            = list(range(50, 601, 50))
p['probe_save_dir']               = '$PROBE_DIR'
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml: checker alpha=0.5 ours_standard=[0.5,0.4,0.3,0.2] probe_save_epochs=50..600', flush=True)
PYEOF

LOG=$OUT_DIR/step5_suboverlap.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
python3 main.py --defense ours 2>&1 | tee $LOG

echo ""
echo "Training done. Client delta files saved:"
ls $PROBE_DIR/client_deltas_ep*.pt 2>/dev/null | wc -l

echo "Running subspace overlap analysis..."
python3 $REPO/jobs/step5_subspace_overlap.py \
    --delta-dir $PROBE_DIR \
    --log       $LOG \
    --out       $OUT_DIR/step5_subspace_overlap.txt \
    2>&1 | tee $OUT_DIR/step5_subspace_overlap_analysis.log

echo "Done."
echo "  Log:     $LOG"
echo "  Overlap: $OUT_DIR/step5_subspace_overlap.txt"
echo "  Deltas:  $PROBE_DIR"
