#!/usr/bin/env bash
#SBATCH -J pal-then-pyr5sp5ev-100k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# Two-mission run: one policy trained on Palisades first, then Pyrenees, with
# the 100000-simulation budget split 50000/50000 between them (--missions with
# equal weights). The missions are phases in the order listed, not interleaved:
# every worker trains Palisades until the fleet has started 50000 simulations
# there, then the whole fleet moves to Pyrenees. Watch for these lines:
#   Mission phase budgets (100000 simulations, weights 1/1), run in this order:
#     Palisades5sp5ev.json -> 50000 simulations then Pyrenees5sp5ev.json -> ...
#   Mission phase complete: Palisades5sp5ev.json finished its 50000 simulations
#   Mission phases actually run: ... 50000/50000 (50.0%), ... 50000/50000
# A phase can overrun its share by roughly the number of workers whose resets
# land in the same step (a handful out of 50000); --simulations caps the total
# at 100000 exactly, so any overrun on Palisades leaves Pyrenees equally short.
#
# 10 aircraft on both maps: 5 example_seaplane_1 + 5 example_aircraft_1 (eVTOL),
# seaplanes listed first in both scenario JSONs, so --group-sizes 5 5 means the
# same thing in both phases -- one tactic decision for all 5 seaplanes and one
# for all 5 eVTOLs, 2 action decisions per step. Palisades puts them on its two
# bases ([5,0] and [0,5]), Pyrenees spreads them over its four ([2,1,1,1] each).
#
# --switch-ignition-4: ignition randomized each episode within a 400x400-cell
# box (2.0x2.0 km ground) around each scenario's own ignition center, with the
# relaxed 150 m urban keep-out. water set 2 (bodies example_aircraft_1 can
# scoop), directional-2 state space (K=3), --cell-size code for the projected
# metres/cell geometry -- the same setup as the single-map palisades_5sp5ev and
# pyrenees_5sp5ev chains, so the phases are comparable to them.
#
# NOTE: multi-mission runs add the 3 scenario one-hot features, so this policy
# has 69 observation features against the 66 of the single-map 5sp5ev runs.
# Checkpoints are therefore NOT interchangeable with those runs; the one-hot is
# what lets one policy tell the two maps apart.
#
# Sizing: measured rates for this exact fleet and config are 2430 simulations/h
# on Palisades (job 677151, 8608 sims in 3.5 h) and 2380/h on Pyrenees (chunks
# 19-21 of the pyrenees_5sp5ev chain, ~8340 sims per 3.5 h). 50000 + 50000
# simulations is therefore ~20.6 h + ~21.0 h = ~42 h. The 96 h wall limit
# (owl_normal_base; owl_normal_short caps at 24 h) leaves better than 2x
# headroom, and --max-train-hours 92 makes the run save and exit on its own if
# the rate turns out far worse, instead of being SIGKILLed.
#
# --chain-total-simulations is set even though nothing is chained: it is what
# switches the LR onto the episode-driven schedule, decaying 5e-04 -> 0 across
# exactly 100000 simulations. Consequence of training in phases: Palisades runs
# at 5.0e-04 down to ~3.1e-04 and Pyrenees continues from there to 0, so the
# second mission trains at a lower LR than the first. That is inherent to a
# sequential split; --mission-weights or a flatter --lr-decay-exponent are the
# knobs if the Pyrenees phase underfits.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --missions "Palisades5sp5ev.json" "Pyrenees5sp5ev.json" \
  --output-dir "$OUTDIR" \
  --save-model missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims100000.zip \
  --progress-file-tag missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3 \
  --chain-total-simulations 100000 \
  --simulations 100000 \
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
