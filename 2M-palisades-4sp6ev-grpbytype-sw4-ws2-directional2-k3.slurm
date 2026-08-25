#!/usr/bin/env bash
#SBATCH -J pal4sp6ev-grptype-sw4-d2
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

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_4sp6ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# 10 aircraft (4 example_seaplane_1 + 6 example_aircraft_1 eVTOL; Palisades has
# 2 bases, seaplanes [4,0] and eVTOLs [0,6] as in every Palisades 4sp*ev).
# --group-sizes 4 6: one tactic decision for all 4 seaplanes and one for all
# 6 eVTOLs -> 2 action decisions per step (grouped by aircraft type). This
# relies on the scenario listing seaplanes before eVTOLs; verified against
# this exact fleet.
# --switch-ignition-4: ignition randomized each episode within a 400x400-cell
# box around the scenario's original ignition center, with a relaxed 150 m
# urban keep-out (vs 350 m for modes 1/2/3) so fires may start nearer the
# urban interface. Measured on this scenario: 85839 candidate cells (vs 96802
# for sw3, whose box is larger but whose keep-out is stricter); over 40 resets
# the ignition displacement from the scenario point is median 641 m, p90
# 1302 m, max 1696 m -- a tighter tail than sw3 (median 661 m, max 2632 m)
# because the smaller box caps the reach even as the keep-out opens up.
# NOTE: as with sw1/sw2/sw3 on Palisades, the residential wedge biases
# ignitions east/southeast of the scenario point.
# water set 2 (bodies example_aircraft_1 can scoop), directional-2 state
# space (K=3): the directional core with the four map-boundary distances
# removed -- each was an exact duplicate of a fire-extreme coordinate
# (distance_left_boundary == leftmost_x, distance_right_boundary ==
# 1 - rightmost_x, and likewise bottom/top) -- and the episode's ignition
# point (ignition_x, ignition_y, scaled to [0,1] on the same map extent as
# fire_center_x/y) added in their place. 66 observation features:
# 28 core + 3*6 flank + 2*10 agent. NOTE: the observation dimension differs
# from the 'directional' runs (68 for this fleet), so checkpoints are not
# interchangeable between the two series.
# --cell-size code uses the actual projected metres/cell for the corrected
# flank geometry (2c/2e/2f) and the rescaled vip/indirect/water channels.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades4sp6ev.json" \
  --num-envs 128 \
  --timesteps 2016000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_4sp6ev_grpbytype_sw4_directional2_k3.zip \
  --progress-file-tag palisades_4sp6ev_grpbytype_sw4_directional2_k3 \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 4 6 \
  --controlled-agent-count 10 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional-2 \
  --cell-size code \
  --checkpoint-interval 504000
