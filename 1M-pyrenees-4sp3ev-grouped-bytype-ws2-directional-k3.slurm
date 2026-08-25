#!/usr/bin/env bash
#SBATCH -J pyr4sp3ev-grp-dir-k3
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# 128 forked envs on a 96-CPU allocation. Cap each worker's numerical
# libraries to one thread so the workers do not over-subscribe the node.
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_4sp3ev_grouped_bytype_directional_k3_${SLURM_JOB_ID}

# 7 aircraft: 4 example_seaplane_1 (agents 0-3) + 3 example_aircraft_1 eVTOL
# (agents 4-6). Grouped tactics: --tactic-distribution group with
# --aircraft-group-size 4 gives ceil(7/4)=2 tactic decisions per step. The
# first decision applies to agents 0-3 (all 4 seaplanes) and the second to
# agents 4-6 (all 3 eVTOLs) -- i.e. one tactic per aircraft type. This relies
# on the scenario ordering seaplanes before eVTOLs and on group_size equal to
# the seaplane count; verified against this exact fleet.
# water set 2 (bodies example_aircraft_1 can scoop), fixed (scenario) ignition,
# directional state space (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected flank
# geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees4sp3ev.json" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_4sp3ev_grouped_bytype_directional_k3.zip \
  --progress-file-tag pyrenees_4sp3ev_grouped_bytype_directional_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution group \
  --aircraft-group-size 4 \
  --controlled-agent-count 7 \
  --water-set 2 \
  --state-space directional \
  --cell-size code
