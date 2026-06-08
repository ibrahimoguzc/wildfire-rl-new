#!/usr/bin/env bash
#SBATCH -J util-replay
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=80G
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

# Keep numerical libraries from over-subscribing the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

REPO_ROOT=$(pwd)
TAG=aircraft_utilization_replay
DECISION_STEPS=${REPO_ROOT}/examples/wildfire/data/scenarios/outputs/aircraft_utilization_check_553616/results_aircraft_utilization_check_605_decision_steps.csv
OUTDIR=${REPO_ROOT}/examples/wildfire/data/scenarios/outputs/${TAG}_${SLURM_JOB_ID}

# Override at submit time with:
# sbatch --export=ALL,MAX_EPISODES=605 utilization_replay.slurm
MAX_EPISODES=${MAX_EPISODES:-20}

mkdir -p "$OUTDIR"

/usr/bin/time -v python -u examples/wildfire/replay_aircraft_utilization.py \
  --decision-steps "$DECISION_STEPS" \
  --scenario "Palisades copy.json" \
  --max-episodes "$MAX_EPISODES" \
  --output-dir "$OUTDIR" \
  --decision-interval-minutes 10 \
  --fire-detection-delay-minutes 30 \
  --state-space mixed \
  --state-fire-fronts 5 \
  --controlled-agent-count 6 \
  --aircraft-group-size 2
