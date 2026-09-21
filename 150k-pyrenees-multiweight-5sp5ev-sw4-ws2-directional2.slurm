#!/usr/bin/env bash
#SBATCH -J pyr-mw-150k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k_${SLURM_JOB_ID}

# Pyrenees5sp5ev_multiweight.json, 150000 simulations, directional-2.
#
# PER-AIRCRAFT-TYPE firefront cost weights at the full 150k budget (the 25k
# Palisades run was the pilot). The scenario carries a top-level "weights"
# array, one entry per agents[] entry, matched by position:
#   type 0  example_seaplane_1   distance/vip/priority/vegetation/topography = 1.0
#   type 1  example_aircraft_1   distance/vip/priority = 1.0,
#                                vegetation/topography = 0.75
# so eVTOLs discount vegetation and topography relative to seaplanes when
# scoring which firefront to attack. Consumer: select_poi.py, tactic layer.
#
# RUNNER: ppo_runnerv3 - v2 plus the _scenario_cost_weights log line and
# nothing else, so training behaviour matches the directional-2 study runs
# and the log records what each aircraft type actually used.
#
# Paired with 150k-{palisades,pyrenees}-multiweight-5sp5ev-sw4-ws2-
# directional2.slurm: identical setup, different map, so the pair extends the
# dedicated-per-map arms of the theater study with per-type weights.
#
# --chain-total-simulations puts the LR on the episode-driven schedule
# (5e-04 -> 0 across exactly 150000 simulations); with only --simulations the
# LR would decay against the timestep ceiling, which episodes never reach.
#
# Sizing: Pyrenees runs ~2100 simulations/h for this fleet, so 150000 is
# ~71 h. 96 h wall (owl_normal_base); --max-train-hours 92 forces a
# clean save instead of a SIGKILL if the rate slips.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv3 \
  --scenario "Pyrenees5sp5ev_multiweight.json" \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_multiweight_5sp5ev_directional2_sims150000.zip \
  --progress-file-tag pyrenees_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_150k \
  --chain-total-simulations 150000 \
  --simulations 150000 \
  --max-train-hours 92 \
  --num-envs 128 \
  --device cpu \
  --vec-start-method fork \
  --state-fire-fronts 3 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --group-sizes 5 5 \
  --controlled-agent-count 10 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional-2 \
  --cell-size code \
  --summary-retention 3 \
  --checkpoint-interval 252000
