#!/usr/bin/env bash
#SBATCH -J pyr4sp10ev-grptype-sw3
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_4sp10ev_grpbytype_sw3_directional_k3_${SLURM_JOB_ID}

# 14 aircraft (4 example_seaplane_1 + 10 example_aircraft_1 eVTOL; seaplanes
# [1,1,1,1] and eVTOLs [3,3,2,2] across the 4 Pyrenees bases). This is the
# 3sp10ev fleet (job 646948) with a 4th seaplane added at base 4.
# --group-sizes 4 10: one tactic decision for all 4 seaplanes and one for all
# 10 eVTOLs -> 2 action decisions per step (grouped by aircraft type). This
# relies on the scenario listing seaplanes before eVTOLs; verified against
# this exact fleet.
# --switch-ignition-3: ignition randomized each episode within a 500x500-cell
# box around the scenario's original ignition center. Cells are 5 m of real
# ground, so this is a 2.5x2.5 km ground box (vs 0.5 km for sw1 and 4.0 km for
# sw2), leaving 3.84 km2 ignitable after excluding water/urban/rock with the
# 350 m urban keep-out. Median ignition displacement ~945 m.
# NOTE: 14 aircraft is the largest fleet run so far (~56% more ABM work per sim
# step than the 9-aircraft runs), which is why this run is held to 1M timesteps.
# --checkpoint-interval writes every 252k steps to $OUTDIR/checkpoints/ so a
# wall-limit kill is resumable.
# water set 2 (bodies example_aircraft_1 can scoop), directional state space
# (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected
# flank geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees4sp10ev.json" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_4sp10ev_grpbytype_sw3_directional_k3.zip \
  --progress-file-tag pyrenees_4sp10ev_grpbytype_sw3_directional_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 4 10 \
  --controlled-agent-count 14 \
  --switch-ignition-3 \
  --water-set 2 \
  --state-space directional \
  --cell-size code \
  --checkpoint-interval 252000
