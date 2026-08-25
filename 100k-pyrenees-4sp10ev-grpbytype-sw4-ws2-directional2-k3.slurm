#!/usr/bin/env bash
#SBATCH -J pyr4sp10ev-100k
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 96:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_base
#SBATCH --mail-type=BEGIN,END,FAIL
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_4sp10ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# 14 aircraft: 4 example_seaplane_1 [1,1,1,1] + 10 example_aircraft_1 (eVTOL)
# [3,3,2,2] over the 4 Pyrenees bases, seaplanes listed first.
#
# The point of this fleet: every Pyrenees run so far has been ordered by
# seaplane count, not aircraft count (7sp+7ev 70.5% escapes, 6sp+6ev 80.4%,
# 5sp+8ev 82.5%, 5sp+5ev 84.4%). 4sp+10ev is the same 14 aircraft as the best
# run but with the seaplanes cut to 4, the fewest tested. If seaplanes really
# dominate it should land at or below the 5-seaplane runs despite the fleet
# size; if it lands near 6sp+6ev then total aircraft matter more than the
# earlier comparisons suggested.
#
# --group-sizes 4 10: grouped by aircraft type, 2 action decisions per step.
# --switch-ignition-4: ignition randomized each episode within a 400x400-cell
# box around the scenario's ignition center, with a relaxed 150 m urban
# keep-out. water set 2 (bodies example_aircraft_1 can scoop), directional-2
# state space (K=3), --cell-size code for the projected metres/cell geometry.
#
# Single job rather than a chain, as requested. Sizing: the other 14-aircraft
# Pyrenees run (7sp+7ev) sustained 1716 simulations/h, so 100000 simulations is
# ~59 h. The 96 h wall limit needs owl_normal_base (owl_normal_short caps at
# 24 h); --max-train-hours 94 makes the run save and exit on its own two hours
# short of that, so a slower-than-expected rate ends in a usable model instead
# of a SIGKILL.
#
# --chain-total-simulations is set even though nothing is chained: it is what
# switches the LR onto the episode-driven schedule, decaying 5e-04 -> 0 across
# exactly 100000 simulations. Without it the LR would decay against the
# timestep ceiling (100000 x steps-per-episode), which episodes never reach
# because they end when the fire escapes, leaving the run finishing at a high
# LR and not comparable to the chained runs.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees4sp10ev.json" \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_4sp10ev_grpbytype_sw4_directional2_k3_sims100000.zip \
  --progress-file-tag pyrenees_4sp10ev_grpbytype_sw4_directional2_k3 \
  --chain-total-simulations 100000 \
  --simulations 100000 \
  --max-train-hours 94 \
  --num-envs 128 \
  --device cpu \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 4 10 \
  --controlled-agent-count 14 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional-2 \
  --cell-size code \
  --summary-retention 3 \
  --checkpoint-interval 252000
