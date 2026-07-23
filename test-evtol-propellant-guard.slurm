#!/usr/bin/env bash
#SBATCH -J test-evtol-guard
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail

module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
mkdir -p logs

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/wildfire-evtol-guard-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${CACHE_ROOT}/numba"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCENARIO="${SCENARIO:-Palisades3sp3dr.json}"
RUNS="${RUNS:-8}"
SEED="${SEED:-0}"

echo "Running focused EVTOL propellant regression tests"
python -m pytest --no-cov -q \
  tests/test_examples/test_wildfire/test_evtol_water_energy.py

echo "Running EVTOL Palisades smoke test"
echo "Scenario: ${SCENARIO}"
echo "Runs: ${RUNS}"
echo "Seed: ${SEED}"
python -u -m examples.wildfire.run_scenarios_n_times \
  --scenarios "${SCENARIO}" \
  --runs "${RUNS}" \
  --seed "${SEED}"

echo "EVTOL propellant guard validation completed successfully"
