#!/usr/bin/env bash
#SBATCH -J test-evtol-fix
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ibrahimoguz@vt.edu

# Validates the eVTOL propellant-guard fixes against the exact config that
# crashed in jobs 594738 / 595134 (Palisades4sp3ev.json, PPO, individual
# tactics). Two prior crashes are covered here:
#   * 594794: Task.__eq__ AttributeError in the step-level divert guard
#             (now identity-compared in agents.py).
#   * 595134: "Agent ran out of electric propellant." -- the eVTOL firefront
#             return-energy guard was clobbered every PPO decision when the
#             policy reassigned tactic.track_poi. The swap now lives in the
#             SuppresionTactic.track_poi setter, so it survives reassignment.
# A SHORT timestep budget is enough: the original crashes hit within the first
# few rollouts. If this clears several rollouts with no ValueError, it holds.

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Per-job caches so concurrent jobs do not contend on numba/matplotlib.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/wildfire-evtol-fix-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# 128 forked envs on a 96-CPU node: cap each worker to one BLAS thread so the
# numerical libraries do not over-subscribe.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Tunables (override on the command line, e.g. TIMESTEPS=200000 sbatch ...).
NUM_ENVS="${NUM_ENVS:-128}"
TIMESTEPS="${TIMESTEPS:-100000}"

echo "=== [1/2] eVTOL propellant regression tests ==="
python -m pytest --no-cov -q \
  tests/test_examples/test_wildfire/test_evtol_water_energy.py

OUTDIR=examples/wildfire/data/scenarios/outputs/evtol_fix_validation_${SLURM_JOB_ID}

echo "=== [2/2] PPO smoke on Palisades4sp3ev.json (num_envs=${NUM_ENVS}, timesteps=${TIMESTEPS}) ==="
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades4sp3ev.json" \
  --num-envs "${NUM_ENVS}" \
  --timesteps "${TIMESTEPS}" \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model evtol_fix_validation.zip \
  --progress-file-tag evtol_fix_validation \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution individual \
  --controlled-agent-count 7 \
  --water-set 2 \
  --state-space directional \
  --cell-size code

echo "=== eVTOL fix validation completed with no propellant crash ==="
