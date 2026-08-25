#!/usr/bin/env bash
#SBATCH -J pal-then-pyr5sp5ev-constlr
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

OUTDIR=examples/wildfire/data/scenarios/outputs/missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_constlr_${SLURM_JOB_ID}

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
# CONSTANT LEARNING RATE. This is the sibling of job 679425 (the decaying-LR
# version) and differs from it in exactly one flag: --lr-decay-exponent 0.
# The schedule is lr(p) = initial * p^exponent, so an exponent of 0 makes
# p^0 = 1 for all p and the LR stays at 5e-04 for the whole 100000
# simulations.
#
# Why: in the decaying run the two phases do not get equal treatment. Palisades
# trains from 5.0e-04 down to ~3.1e-04, then Pyrenees continues from ~3.1e-04
# to 0 - so the second mission, which is by far the harder map, learns at a
# systematically lower LR than the first and finishes with almost no step size
# left. Early evidence from 679425 is consistent with that hurting: 6089
# simulations into its Pyrenees phase it sat at 95.4% escapes, worse than the
# ~94% a from-scratch Pyrenees run shows at the same point. Holding the LR flat
# removes the confound, so any remaining Palisades-to-Pyrenees transfer effect
# is attributable to the curriculum rather than to the schedule.
#
# Cost of the choice: nothing anneals, so the final policy will be noisier than
# an annealed one and its last-few-thousand-simulation average will understate
# what the weights can do. That is the accepted trade for a clean comparison;
# read this run against 679425 phase-by-phase, not as a bid for the best
# absolute score.
#
# --chain-total-simulations is still passed so every other aspect matches
# 679425 (it selects the episode-driven schedule and reports progress); with
# exponent 0 it no longer affects the LR value.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --missions "Palisades5sp5ev.json" "Pyrenees5sp5ev.json" \
  --output-dir "$OUTDIR" \
  --save-model missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_constlr_sims100000.zip \
  --progress-file-tag missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_constlr \
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
  --lr-decay-exponent 0 \
  --gamma 0.99 \
  --group-sizes 5 5 \
  --controlled-agent-count 10 \
  --switch-ignition-4 \
  --water-set 2 \
  --state-space directional-2 \
  --cell-size code \
  --summary-retention 3 \
  --checkpoint-interval 252000
