#!/bin/bash
#SBATCH --job-name=step5_ode
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuL
#SBATCH --array=1-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --exclude=node871,node877

# ODE probe: measure m(t), proj_adv, proj_ben at the real normcap scale (eps=0.057).
# Tasks: 1=checker seed1, 2=checker seed2, 3=blend_87 seed1, 4=blend_87 seed2.
# Logs per round: [ODE_INIT], [ODE_PROJ], [ODE_M].
# Summary computes: dm_obs, dm_pred, and per-round attacker/benign contributions.
# Goal: check if benign_term ~0 (checker) vs <0 (blend_87) in the linear regime.

if   [[ $SLURM_ARRAY_TASK_ID -le 2 ]]; then
    TRIGGER=checker
    SEED=$SLURM_ARRAY_TASK_ID
else
    TRIGGER=blend_87
    SEED=$(( SLURM_ARRAY_TASK_ID - 2 ))
fi
TARGET_NORM=0.057
ROUNDS=600
AMPLITUDE=0.1145

echo "step5_ode job $SLURM_JOB_ID arm $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM"

SCRATCH=/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch
SCRATCH_DIR=$SCRATCH/step5_ode_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $SCRATCH_DIR

OUT_DIR=$HOME/ffl_step5
mkdir -p $OUT_DIR

REPO=$HOME/tba_attack_test
VENV=$HOME/venvs/tba_attack
DATA_DIR=$HOME/filterfl_test/filterfl_test/data/cifar-10-batches-py
ROOT_PKL=$OUT_DIR/fltrust_root.pkl
PYTHON=/opt/apps/el9-fix/apps/binapps/anaconda3/2024.10/bin/python3.12

[[ -x "$PYTHON" ]] || { echo "FATAL: python not found"; exit 1; }
export PYTHONPATH=$VENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}

cp -r $REPO/FilterFL $SCRATCH_DIR/
mkdir -p $SCRATCH_DIR/FilterFL/data
cp -r $DATA_DIR $SCRATCH_DIR/FilterFL/data/ 2>/dev/null && echo "CIFAR-10 staged"

# ── Patch 1: normcap + ODE probe (per-round projections) ─────────────────────
export SCRATCH_DIR TARGET_NORM ODE_TRIGGER=$TRIGGER ODE_AMPLITUDE=$AMPLITUDE
$PYTHON -I - <<'PATCHEOF'
import os, sys, py_compile

scratch     = os.environ['SCRATCH_DIR']
tn          = float(os.environ['TARGET_NORM'])
alg_path    = f'{scratch}/FilterFL/algorithm.py'

with open(alg_path) as f:
    src = f.read()

patch = f'''\
        # [NORMCAP+ODE_PROBE]
        import numpy as _np_nc, os as _os_nc, sys as _sys_nc
        import pickle as _pkl_nc, copy as _copy_nc
        _lnc_target = {tn}
        _lnc_keys   = [k for k in ori_weight if ori_weight[k].dtype != torch.int64]
        _pkeys_ode  = [k for k in ori_weight
                       if ori_weight[k].dtype != torch.int64
                       and 'running_' not in k and 'num_batches' not in k]

        _adv_idx = [i for i,n in enumerate(agent_name_keys) if n in adversarial_name_keys]
        _ben_idx = [i for i,n in enumerate(agent_name_keys) if n not in adversarial_name_keys]

        # ── One-time setup: triggered test images + v_m ──────────────────────
        if not hasattr(handle, '_ode_vm_unit'):
            _data_file = '{scratch}/FilterFL/data/cifar-10-batches-py/test_batch'
            with open(_data_file, 'rb') as _fh:
                _td = _pkl_nc.load(_fh, encoding='bytes')
            _imgs_r = _td[b'data'].reshape(-1,3,32,32).astype(_np_nc.float32)[:500]/255.0
            _MEAN = _np_nc.array([0.4914,0.4822,0.4465],dtype=_np_nc.float32).reshape(3,1,1)
            _STD  = _np_nc.array([0.2023,0.1994,0.2010],dtype=_np_nc.float32).reshape(3,1,1)
            _imgs_n = (_imgs_r - _MEAN) / _STD
            _sys_nc.path.insert(0, '{scratch}/FilterFL')
            from trigger_utils import make_raw_delta as _mrd
            _delta = _mrd(_os_nc.environ['ODE_TRIGGER'],
                          float(_os_nc.environ['ODE_AMPLITUDE']))
            _imgs_trig_np = _np_nc.clip(_imgs_n + _delta/_STD, -4.0, 4.0)
            _dev = next(handle.model.parameters()).device
            _imgs_trig_t  = torch.tensor(_imgs_trig_np).to(_dev)
            # v_m: gradient of mean target margin (airplane=0) w.r.t. model params
            _vm_m = _copy_nc.deepcopy(handle.model)
            _vm_m.train(); _vm_m.zero_grad()
            _OTH = [c for c in range(10) if c != 0]
            for _bi in range(0, len(_imgs_trig_t), 128):
                _xb = _imgs_trig_t[_bi:_bi+128]
                _lg = _vm_m(_xb)
                (_lg[:, 0] - _lg[:, _OTH].max(1).values).mean().backward()
            _vm_flat = _np_nc.concatenate([
                p.grad.float().cpu().numpy().flatten()
                for p in _vm_m.parameters()])
            _vm_nrm  = float(_np_nc.linalg.norm(_vm_flat))
            _vm_unit = _vm_flat / (_vm_nrm + 1e-9)
            del _vm_m
            handle._ode_imgs_t  = _imgs_trig_t
            handle._ode_vm_unit = _vm_unit
            handle._ode_vm_norm = _vm_nrm
            handle._ode_OTHER   = _OTH
            logger.info(f'[ODE_INIT] ep={{epoch}} ||v_m||={{_vm_nrm:.4f}}')

        # ── Per-round: project each client delta onto v_m_unit ────────────────
        _vm_u = handle._ode_vm_unit
        _N    = len(agent_name_keys)
        _adv_proj, _ben_proj = 0.0, 0.0
        for _ii, _nn in enumerate(agent_name_keys):
            _d = _np_nc.concatenate([
                (w_locals[_ii][k].float().cpu().numpy()
                 - ori_weight[k].float().cpu().numpy()).flatten()
                for k in _pkeys_ode])
            _p = float(_np_nc.dot(_d, _vm_u)) / _N   # equal-weight FedAvg approx
            if _nn in adversarial_name_keys:
                _adv_proj += _p
            else:
                _ben_proj += _p
        logger.info(f'[ODE_PROJ] ep={{epoch}} proj_adv={{_adv_proj:.6f}} '
                    f'proj_ben={{_ben_proj:.6f}} proj_tot={{_adv_proj+_ben_proj:.6f}}')

        # ── Normcap ──────────────────────────────────────────────────────────
        for _lnc_i, _lnc_name in enumerate(agent_name_keys):
            if _lnc_name in adversarial_name_keys:
                _sq = sum((w_locals[_lnc_i][k] - ori_weight[k]).float().pow(2).sum().item()
                          for k in _lnc_keys)
                _nrm = _sq ** 0.5
                if _nrm > _lnc_target:
                    _sc = _lnc_target / _nrm
                    for k in _lnc_keys:
                        w_locals[_lnc_i][k] = (
                            ori_weight[k] + (w_locals[_lnc_i][k] - ori_weight[k]) * _sc
                        ).to(w_locals[_lnc_i][k].dtype)
        logger.info(f'[CLIENT_FLAGS] ep={{epoch}} '
                    f'order=[{{",".join(str(n) for n in agent_name_keys)}}] '
                    f'adv_mask=[{{",".join("1" if n in adversarial_name_keys else "0" for n in agent_name_keys)}}]')
        # [END NORMCAP+ODE_PROBE]
'''

marker = '        # ── FLAME / FLTrust aggregation'
if marker not in src:
    print('ERROR: marker not found'); sys.exit(1)
src = src.replace(marker, patch + marker, 1)

# ── Patch 2: log m(t) after global model update ───────────────────────────────
m_patch = '''
        # [ODE_M] Target margin on triggered test inputs after aggregation
        if hasattr(handle, '_ode_vm_unit'):
            import numpy as _np_m
            handle.model.eval()
            _all_m = []
            with torch.no_grad():
                for _bi in range(0, len(handle._ode_imgs_t), 128):
                    _xb = handle._ode_imgs_t[_bi:_bi+128]
                    _lg = handle.model(_xb)
                    _lt  = _lg[:, 0]
                    _lmx = _lg[:, handle._ode_OTHER].max(1).values
                    _all_m.append((_lt - _lmx).cpu().numpy())
            _m_t = float(_np_m.concatenate(_all_m).mean())
            logger.info(f'[ODE_M] ep={epoch} m_t={_m_t:.4f}')
        # [END ODE_M]
'''

ld_marker = '        handle.model.load_state_dict(w_glob)'
if ld_marker not in src:
    print('ERROR: load_state_dict marker not found'); sys.exit(1)
src = src.replace(ld_marker, ld_marker + '\n' + m_patch, 1)

with open(alg_path, 'w') as f:
    f.write(src)

try:
    py_compile.compile(alg_path, doraise=True)
    print(f'patch applied (cap={tn}): compiles OK')
except py_compile.PyCompileError as e:
    print(f'compile error: {e}'); sys.exit(1)
PATCHEOF
if [[ $? -ne 0 ]]; then echo "Patch failed"; exit 1; fi

# ── YAML ─────────────────────────────────────────────────────────────────────
$PYTHON -I - <<PYEOF
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
p['epochs']                       = $ROUNDS
p['seed']                         = $SEED
p['dirichlet_alpha']              = 0.5
p['ours_standard']                = [0.5, 0.4, 0.3, 0.2]
p['trigger_type']                 = '$TRIGGER'
p['trigger_amplitude_raw']        = $AMPLITUDE
p['probe_save_epochs']            = []
p['fltrust_root_path']            = '$ROOT_PKL'
p['log_ben_s3']                   = False
with open('$SCRATCH_DIR/FilterFL/cifar_params.yaml', 'w') as f:
    yaml.dump(p, f)
print('yaml ok trigger=$TRIGGER seed=$SEED cap=$TARGET_NORM', flush=True)
PYEOF

LOG=$OUT_DIR/step5_ode_${TRIGGER}_seed${SEED}.log

cd $SCRATCH_DIR/FilterFL
mkdir -p saved_models
$PYTHON main.py --defense ours 2>&1 | tee $LOG

echo "Training done."

# ── Summary: extract m(t), proj_adv, proj_ben, validate ODE ──────────────────
$PYTHON -I -W ignore - <<SUMEOF
import re, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/venvs/tba_attack/lib/python3.12/site-packages')
import numpy as np

seed    = $SEED
trigger = '$TRIGGER'
CTRL    = 0.019
LOG_PATH = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ode_{trigger}_seed{seed}.log'
OUT_FILE = f'/mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/ffl_step5/step5_ode_{trigger}_seed{seed}_summary.txt'

with open(LOG_PATH) as f:
    LOG = f.read()

# m(t) trajectory
mt_raw = re.findall(r'\[ODE_M\] ep=(\d+) m_t=([+-]?[\d.eE+-]+)', LOG)
mt = {int(ep): float(m) for ep, m in mt_raw}

# projections
proj_raw = re.findall(
    r'\[ODE_PROJ\] ep=(\d+) proj_adv=([+-]?[\d.eE+-]+) proj_ben=([+-]?[\d.eE+-]+) proj_tot=([+-]?[\d.eE+-]+)',
    LOG)
projs = {int(ep): (float(pa), float(pb), float(pt)) for ep, pa, pb, pt in proj_raw}

# v_m norm
vm_init = re.search(r'\[ODE_INIT\].*\|\|v_m\|\|=([+-]?[\d.eE+-]+)', LOG)
vm_norm = float(vm_init.group(1)) if vm_init else float('nan')

# ASR trajectory (for reference)
evals = re.findall(r'___Global_Test_Poison:(False|True).*?Accuracy: \d+/\d+ \(([\d.]+)%\)', LOG)
pairs, i = [], 0
while i+1 < len(evals):
    if evals[i][0]=='False' and evals[i+1][0]=='True':
        pairs.append((float(evals[i][1])/100, float(evals[i+1][1])/100))
    i += 2

lines = [f'=== ODE probe: {trigger} seed={seed} ===', '']
lines.append(f'  ||v_m|| = {vm_norm:.4f}')

if mt:
    eps = sorted(mt.keys())
    m_nat_est = np.mean([mt[e] for e in eps[-50:]])  # mean of last 50 rounds ~ m_nat
    lines.append(f'  m_nat (last-50 mean): {m_nat_est:.4f}')
    lines.append(f'  Rounds with m(t) logged: {len(mt)}')
    lines.append('')

    # ODE decomposition: for each round, dm_obs vs dm_pred
    # dm_pred = (proj_adv + proj_ben) * ||v_m||  (first-order Taylor)
    dm_obs_list, dm_pred_list = [], []
    proj_adv_list, proj_ben_list = [], []
    for ep in sorted(eps):
        if ep-1 in mt and ep in projs:
            dm_obs = mt[ep] - mt[ep-1]
            pa, pb, pt = projs[ep]
            dm_pred = pt * vm_norm
            dm_obs_list.append(dm_obs); dm_pred_list.append(dm_pred)
            proj_adv_list.append(pa * vm_norm)
            proj_ben_list.append(pb * vm_norm)

    lines.append('  ODE decomposition (dm = attacker + benign, first-order Taylor):')
    lines.append(f'  dm_obs   mean={np.mean(dm_obs_list):.4f}  std={np.std(dm_obs_list):.4f}')
    lines.append(f'  dm_pred  mean={np.mean(dm_pred_list):.4f}  std={np.std(dm_pred_list):.4f}')
    lines.append(f'  proj_adv*||v_m||: mean={np.mean(proj_adv_list):.4f}  std={np.std(proj_adv_list):.4f}')
    lines.append(f'  proj_ben*||v_m||: mean={np.mean(proj_ben_list):.4f}  std={np.std(proj_ben_list):.4f}')
    lines.append(f'  ratio attacker/|benign|: {abs(np.mean(proj_adv_list)/(np.mean(proj_ben_list)+1e-9)):.4f}')
    lines.append('')

    # m(t) trajectory (every 50 rounds)
    lines.append('  m(t) trajectory (every 50 rounds):')
    for ep in sorted(eps)[::50] + [sorted(eps)[-1]]:
        pa_sc = projs.get(ep, (0,0,0))[0] * vm_norm
        pb_sc = projs.get(ep, (0,0,0))[1] * vm_norm
        asr = pairs[ep-1][1] if ep-1 < len(pairs) else float('nan')
        lines.append(f'    ep={ep:4d}  m={mt[ep]:+7.3f}  adv={pa_sc:+7.4f}  ben={pb_sc:+7.4f}  asr={asr:.3f}')

out = '\n'.join(lines)
print(out)
with open(OUT_FILE, 'w') as f:
    f.write(out + '\n')
print(f'Saved: {OUT_FILE}')
SUMEOF

echo "Done. Log: $LOG"
