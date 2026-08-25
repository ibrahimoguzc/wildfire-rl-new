#!/usr/bin/env bash
#SBATCH -J pal4sp4ev-grp4ev-sw1
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

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_4sp4ev_grp4ev_sw1_directional_k3_${SLURM_JOB_ID}

# 8 aircraft (4 example_seaplane_1 + 4 example_aircraft_1 eVTOL).
# --group-sizes 1 1 1 1 4: each seaplane gets its own tactic decision and all
# four eVTOLs share one -> 5 action decisions per step.
# --switch-ignition-1: ignition randomized each episode within a 100x100-cell
# box around the scenario's original ignition center (excludes water/urban).
# water set 2 (bodies example_aircraft_1 can scoop), directional state (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected
# flank geometry and the rescaled vip/indirect/water channels.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades4sp4ev.json" \
  --num-envs 128 \
  --timesteps 2016000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_4sp4ev_grp4ev_sw1_directional_k3.zip \
  --progress-file-tag palisades_4sp4ev_grp4ev_sw1_directional_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 1 1 1 1 4 \
  --controlled-agent-count 8 \
  --switch-ignition-1 \
  --water-set 2 \
  --state-space directional \
  --cell-size code
