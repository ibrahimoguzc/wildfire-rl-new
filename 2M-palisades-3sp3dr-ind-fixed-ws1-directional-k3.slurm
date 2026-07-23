#!/usr/bin/env bash
#SBATCH -J pal3sp3dr-dir-k3-fix
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

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_3sp3dr_ind_directional_k3_fixed_${SLURM_JOB_ID}

# 6 planes (3 example_seaplane_1 + 3 example_aircraft_1), individual tactics,
# water set 2 (bodies example_aircraft_1 can scoop), fixed (scenario) ignition,
# directional state space (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected flank
# geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
# Palisades3sp3dr.json defines 6 aircraft.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades3sp3dr.json" \
  --num-envs 128 \
  --timesteps 2016000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_3sp3dr_ind_directional_k3_fixed.zip \
  --progress-file-tag palisades_3sp3dr_ind_directional_k3_fixed \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution individual \
  --controlled-agent-count 6 \
  --water-set 2 \
  --state-space directional \
  --cell-size code
