#!/usr/bin/env bash
#SBATCH -J abm-profile
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=120G
#SBATCH --qos=owl_normal_short

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# The profiler is a single-process diagnostic run. Keep BLAS/OpenMP libraries
# from spawning extra threads that would distort ABM timing.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT=$(pwd)
OUTROOT=${REPO_ROOT}/examples/wildfire/data/scenarios/outputs/abm_profile_${SLURM_JOB_ID}

# Override at submit time if you want a longer sample:
# sbatch --export=ALL,DECISION_STEPS=5,SEEDS="101 202" abm_profile.slurm
DECISION_STEPS=${DECISION_STEPS:-3}
SEEDS=${SEEDS:-101}
STATE_SPACE=${STATE_SPACE:-small}
STATE_FIRE_FRONTS=${STATE_FIRE_FRONTS:-5}
TOP_N=${TOP_N:-20}

mkdir -p "$OUTROOT"

run_profile() {
  local label="$1"
  local scenario="$2"
  local controlled="$3"
  local distribution="$4"
  local group_size="$5"
  local outdir="${OUTROOT}/${label}"

  mkdir -p "$outdir"
  /usr/bin/time -v python -u examples/wildfire/profile_abm_breakdown.py \
    --scenarios "$scenario" \
    --seeds $SEEDS \
    --decision-steps "$DECISION_STEPS" \
    --decision-interval-minutes 10 \
    --fire-detection-delay-minutes 30 \
    --state-space "$STATE_SPACE" \
    --state-fire-fronts "$STATE_FIRE_FRONTS" \
    --tactic-distribution "$distribution" \
    --aircraft-group-size "$group_size" \
    --controlled-agent-count "$controlled" \
    --action-mode random \
    --top-n "$TOP_N" \
    --output-dir "$outdir"
}

run_profile palisades "Palisades copy.json" 6 group 2
run_profile pyrenees "Pyrenees.json" 12 group 2
run_profile salamis "Salamis.json" 3 individual 2

echo "ABM profile outputs written to: $OUTROOT"
