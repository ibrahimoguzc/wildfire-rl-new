#!/usr/bin/env bash
#SBATCH -J test-evtol-pyr
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

# Validates the eVTOL base-resupply re-energize fix against Pyrenees, the
# scenario that still depleted after the takeoff fix (job 637699).
#
# Pyrenees water is >= 2100 m from every airport (median 15 km), so
# `select_suppressant_source` usually picks BASE. The resupply hold used to
# refill payload only, so the eVTOL parked on the apron, topped up water and
# departed on a draining battery -- repeatedly -- until it ran dry.
# `initiate_resupply` now re-energizes when the chosen destination is a BASE.
# Restricted to eVTOLs: applying it fleet-wide measurably changed conventional
# aircraft here (suppressions 171 -> 147 on seed 0).
#
# The heuristic runner does not reproduce the crash -- it needs the PPO
# policy's tactic churn -- so this runs PPO with a short timestep budget.
# Crash-only diagnostics stay armed: on failure grep the .err for
# EVTOL_PROPELLANT_DEPLETION and the EVTOL_HIST sortie history.

set -euo pipefail

mkdir -p logs

module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/wildfire-evtol-pyr-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# 128 forked envs on a 96-CPU node: one BLAS thread per worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

NUM_ENVS="${NUM_ENVS:-128}"
TIMESTEPS="${TIMESTEPS:-100000}"

echo "=== [1/3] eVTOL propellant regression tests ==="
python -m pytest --no-cov -q \
  tests/test_examples/test_wildfire/test_evtol_water_energy.py

echo "=== [2/3] Pyrenees heuristic missions (2 seeds) ==="
python -u -m examples.wildfire.run_scenarios_n_times \
  --scenarios Pyrenees4sp3ev.json --runs 2 --seed 0

OUTDIR=examples/wildfire/data/scenarios/outputs/evtol_pyr_validation_${SLURM_JOB_ID}

echo "=== [3/3] PPO smoke on Pyrenees4sp3ev (num_envs=${NUM_ENVS}, timesteps=${TIMESTEPS}) ==="
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees4sp3ev.json" \
  --num-envs "${NUM_ENVS}" \
  --timesteps "${TIMESTEPS}" \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model evtol_pyr_validation.zip \
  --progress-file-tag evtol_pyr_validation \
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

echo "=== Pyrenees eVTOL validation completed with no propellant crash ==="
