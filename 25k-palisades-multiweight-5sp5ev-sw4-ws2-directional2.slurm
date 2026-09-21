#!/usr/bin/env bash
#SBATCH -J pal-mw-directional2
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail
mkdir -p logs
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_multiweight_5sp5ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# Palisades5sp5ev_multiweight.json, 25000 simulations, directional-2.
#
# Tests the new PER-AIRCRAFT-TYPE firefront cost weights. The scenario adds a
# top-level "weights" array with one entry per agents[] entry:
#   type 0  example_seaplane_1  distance/vip/priority/vegetation/topography = 1.0
#   type 1  example_aircraft_1  distance=1.0 vip=1.0, priority/vegetation/
#                               topography = 0.5
# so eVTOLs discount priority, vegetation and topography relative to seaplanes
# when scoring which firefront to attack. Verified before submitting that
# WildfireParameters.firefront_cost_weights() resolves those per type; the
# consumer is select_poi.py:276, in the tactic layer.
#
# RUNNER: ppo_runnerv3, not v2. Diffed the two - v3 is v2 plus 74 added lines
# and ZERO removed, all of them _scenario_cost_weights /
# _print_scenario_cost_weights and their call site. So training behaviour is
# identical to v2, and the run log gains a "Firefront cost weights [...]" line
# recording what each aircraft type actually used. Worth having when the whole
# point of the run is that those weights differ; without it nothing in the log
# would distinguish this from a standard Palisades run.
#
# --chain-total-simulations is set although nothing is chained: it puts the LR
# on the episode-driven schedule so it decays 5e-04 -> 0 across exactly 25000
# simulations. With only --simulations the LR would decay against the timestep
# ceiling, which episodes never reach.
#
# Sizing: Palisades runs ~1800 simulations/h for this fleet, so 25000 is ~14 h.
# The 24 h wall (owl_normal_short) covers a slow node; --max-train-hours 22
# forces a clean save rather than a SIGKILL if the rate slips badly.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv3 \
  --scenario "Palisades5sp5ev_multiweight.json" \
  --output-dir "$OUTDIR" \
  --save-model palisades_multiweight_5sp5ev_directional2_sims25000.zip \
  --progress-file-tag palisades_multiweight_5sp5ev_grpbytype_sw4_directional2_k3 \
  --chain-total-simulations 25000 \
  --simulations 25000 \
  --max-train-hours 22 \
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
