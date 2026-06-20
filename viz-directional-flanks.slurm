#!/usr/bin/env bash
#SBATCH -J viz-dir-flanks
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=32G
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

# Snapshot the K-sector directional flank decomposition (2c/2e/2f) every 10
# simulated minutes of a live episode. The script's mirror of
# _compute_directional_front_block is cross-checked against the env every step,
# so a "WARNING: mirrored flank block diverges" line in the log means the
# visualization drifted from what the agent observes.
#
# Pyrenees.json now defines 9 aircraft, Palisades.json defines 4, so
# --controlled-agent-count is set per scenario. --sim-seed fixes the fire for
# reproducibility. Each configuration writes to its own subdir so the per-step
# filenames do not collide.
OUTROOT=examples/wildfire/snapshots/flanks_${SLURM_JOB_ID}

# 1) Pyrenees, corrected geometry (--cell-size code): the configuration the
#    directional fixes are meant to run with.
python -u scripts/visualize_directional_flanks.py \
  --scenario "Pyrenees.json" \
  --state-fire-fronts 4 \
  --steps 36 \
  --sim-seed 42 \
  --controlled-agent-count 6 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --water-set 1 \
  --cell-size code \
  --output-dir "${OUTROOT}/pyrenees_k4_code"

# 2) Pyrenees, nominal cell size (--cell-size json): same fire, legacy 5 m
#    spacing, for a before/after comparison of the projection geometry.
python -u scripts/visualize_directional_flanks.py \
  --scenario "Pyrenees.json" \
  --state-fire-fronts 4 \
  --steps 36 \
  --sim-seed 42 \
  --controlled-agent-count 6 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --water-set 1 \
  --cell-size json \
  --output-dir "${OUTROOT}/pyrenees_k4_json"

# 3) Palisades, corrected geometry. Palisades.json defines 4 aircraft.
python -u scripts/visualize_directional_flanks.py \
  --scenario "Palisades.json" \
  --state-fire-fronts 4 \
  --steps 36 \
  --sim-seed 42 \
  --controlled-agent-count 4 \
  --tactic-distribution individual \
  --water-set 1 \
  --cell-size code \
  --output-dir "${OUTROOT}/palisades_k4_code"

echo "Snapshots written under ${OUTROOT}"
