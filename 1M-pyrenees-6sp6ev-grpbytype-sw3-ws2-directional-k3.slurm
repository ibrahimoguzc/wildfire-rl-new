#!/usr/bin/env bash
#SBATCH -J pyr6sp6ev-grptype-sw3
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_6sp6ev_grpbytype_sw3_directional_k3_${SLURM_JOB_ID}

# 12 aircraft (6 example_seaplane_1 + 6 example_aircraft_1 eVTOL; both types are
# distributed [2,2,1,1] across the 4 bases).
# --group-sizes 6 6: one tactic decision for all 6 seaplanes and one for all
# 6 eVTOLs -> 2 action decisions per step (grouped by aircraft type). This
# relies on the scenario listing seaplanes before eVTOLs; verified against
# this exact fleet.
# --switch-ignition-3: ignition randomized each episode within a 500x500-cell
# box around the scenario's original ignition center. Cells are 5 m of real
# ground, so this is a 2.5x2.5 km ground box (vs 0.5 km for sw1 and 4.0 km for
# sw2), leaving 3.84 km2 ignitable after excluding water/urban/rock with the
# 350 m urban keep-out. Median ignition displacement ~945 m.
# water set 2 (bodies example_aircraft_1 can scoop), directional state space
# (K=3).
# --cell-size code uses the actual projected metres/cell for the corrected
# flank geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
# 1M timesteps to match the other Pyrenees sw3 runs (12 aircraft is ~33% more
# ABM work per sim step than the 10-aircraft 4sp6ev run, so expect a lower
# throughput; the 24 h wall clock plus checkpoints cover a short fall.)
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees6sp6ev.json" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_6sp6ev_grpbytype_sw3_directional_k3.zip \
  --progress-file-tag pyrenees_6sp6ev_grpbytype_sw3_directional_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 6 6 \
  --controlled-agent-count 12 \
  --switch-ignition-3 \
  --water-set 2 \
  --state-space directional \
  --cell-size code \
  --checkpoint-interval 252000
