#!/usr/bin/env bash
#SBATCH -J pyr10p-grpbase-sw4
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_10p_grpbybase_sw4_directional_k3_${SLURM_JOB_ID}

# 10 aircraft, all example_seaplane_1, [3,3,2,2] across the 4 Pyrenees bases.
# Seaplane-only fleet, so the scenario lists a single agent type (the *p naming
# convention used by Pyrenees4p/5p/6p/7p) rather than a seaplane+eVTOL pair.
# --group-sizes 3 3 2 2: one group per base -> 4 action decisions per step.
# Each group shares both a tactic and a home base, matching the eVTOL grouping
# used in 1M-pyrenees-4sp12ev-grp4x4. Grouping by aircraft TYPE would collapse
# to a single group here (only seaplanes present, 1 decision), which is what
# job 650281 (Pyrenees7p) runs -- so this job is NOT a clean fleet-size pair
# with 650281: it differs in both fleet size and decision count.
# --switch-ignition-4: ignition randomized each episode within a 400x400-cell
# box around the scenario's original ignition center. Cells are 5 m of real
# ground, so this is a 2.0x2.0 km ground box, with a relaxed 150 m urban
# keep-out (vs 350 m for modes 1/2/3), leaving 3.15 km2 ignitable on Pyrenees.
# Median ignition displacement ~760 m; ~18% of ignitions fall inside the 350 m
# ring that modes 1/2/3 exclude, so fires start nearer the urban interface.
# water set 2 (bodies example_aircraft_1 can scoop), directional state space
# (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected
# flank geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees10p.json" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_10p_grpbybase_sw4_directional_k3.zip \
  --progress-file-tag pyrenees_10p_grpbybase_sw4_directional_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 3 3 2 2 \
  --controlled-agent-count 10 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional \
  --cell-size code \
  --checkpoint-interval 252000
