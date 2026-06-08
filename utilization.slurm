#!/usr/bin/env bash
#SBATCH -J pal-util
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

# Keep NumPy/PyTorch worker libraries from over-subscribing the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

SCENARIO="Palisades copy.json"
TIMESTEPS=6144
NUM_ENVS=1
STATE_SPACE=mixed
STATE_FIRE_FRONTS=5
CONTROLLED_AGENT_COUNT=6
AIRCRAFT_GROUP_SIZE=2
FIRE_DETECTION_DELAY_MINUTES=30
TAG=aircraft_utilization_check
REPO_ROOT=$(pwd)
OUTDIR=${REPO_ROOT}/examples/wildfire/data/scenarios/outputs/${TAG}_${SLURM_JOB_ID}

mkdir -p "$OUTDIR"

/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "$SCENARIO" \
  --timesteps "$TIMESTEPS" \
  --num-envs "$NUM_ENVS" \
  --device cpu \
  --state-space "$STATE_SPACE" \
  --state-fire-fronts "$STATE_FIRE_FRONTS" \
  --tactic-distribution group \
  --aircraft-group-size "$AIRCRAFT_GROUP_SIZE" \
  --controlled-agent-count "$CONTROLLED_AGENT_COUNT" \
  --fire-detection-delay-minutes "$FIRE_DETECTION_DELAY_MINUTES" \
  --output-dir "$OUTDIR" \
  --progress-file-tag "$TAG" \
  --log-output "$OUTDIR/${TAG}_steps.csv" \
  --summary-output "$OUTDIR/${TAG}_summary.csv"
