#!/usr/bin/env bash
#SBATCH -J pyr-then-pal5sp5ev-200k
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 144:00:00
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

OUTDIR=examples/wildfire/data/scenarios/outputs/missions_pyrenees_palisades_5sp5ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# Two-mission run in the HARD-FIRST order: 100000 simulations on Pyrenees, then
# 100000 on Palisades, one policy, 200000 total (--missions with equal weights).
# Phases are sequential, not interleaved: the whole fleet finishes Pyrenees
# before any worker sees Palisades. Watch for:
#   Mission phase budgets (200000 simulations, weights 1/1), run in this order:
#     Pyrenees5sp5ev.json -> 100000 simulations then Palisades5sp5ev.json -> ...
#
# WHY THIS ORDER. Job 679425 ran the opposite order (Palisades then Pyrenees)
# and the curriculum actively hurt: 47789 simulations into its Pyrenees phase it
# sat at 93.0% escapes / -0.103 MoE against 91.0% / -0.072 for a from-scratch
# Pyrenees run over the same window - a 6.8 sigma deficit that widened rather
# than closed. The hypothesis this run tests is that the order was backwards:
# Palisades saturates at ~3% escapes, so a policy trained there first learns a
# map where almost nothing is at stake and then has to unlearn it. Starting on
# Pyrenees, where the gradient lives, and finishing on Palisades asks the
# cheaper question - can a Pyrenees-competent policy retain that competence
# while picking up an easy map?
#
# The two phases are NOT symmetric measurements. The Pyrenees phase is directly
# comparable to the from-scratch runs (679348 chain, 679379) because it starts
# from scratch too; the Palisades phase measures retention and is expected to
# reach Palisades' usual ~2-3% escapes quickly. The interesting number at the
# end is what the FINAL policy scores back on Pyrenees, which needs a separate
# ppo_tester evaluation - the training curve alone will not show forgetting,
# because after the switch it only ever reports Palisades episodes.
#
# NOT --switch-scenarios: that flag fixes one scenario per worker in round-robin
# and trains both maps concurrently for the whole run (ppo_runnerv2.py ~line
# 434), so it cannot express an order. --missions is the sequential mechanism.
#
# 10 aircraft on both maps: 5 example_seaplane_1 + 5 example_aircraft_1 (eVTOL),
# seaplanes listed first in both scenario JSONs, so --group-sizes 5 5 means the
# same thing in both phases. Pyrenees spreads them over four bases ([2,1,1,1]
# each), Palisades over its two ([5,0] and [0,5]).
#
# --switch-ignition-4, water set 2, directional-2 (K=3), --cell-size code:
# identical to 679425/684399 and to the single-map chains, so the phases stay
# comparable to all of them.
#
# NOTE: multi-mission runs add the scenario one-hot features, so this policy's
# observation vector does not match the single-map 5sp5ev checkpoints. Its
# checkpoints are not interchangeable with theirs.
#
# LR: the default decaying schedule (5e-04 -> 0 over 200000), matching 679425 so
# the order comparison is like-for-like. That does mean the Palisades phase runs
# at a lower LR than the Pyrenees phase - here that is the benign direction,
# since Palisades is the easy map and the phase is a retention check. Job 684399
# is separately measuring what a constant LR does to a two-mission run; if it
# shows a large effect, rerun this with --lr-decay-exponent 0.
#
# Sizing: measured 2050 simulations/h on Pyrenees for this fleet (job 679348
# chain, 96238 sims in 49.3 h and accelerating) and 2430/h on Palisades (job
# 677151). 100000 + 100000 is therefore ~49 h + ~41 h = ~90 h. A 96 h wall would
# leave only 6 h of margin, so this asks for 144 h (owl_normal_base allows up to
# 7 days) with --max-train-hours 140 to save and exit cleanly if the rate slips.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --missions "Pyrenees5sp5ev.json" "Palisades5sp5ev.json" \
  --output-dir "$OUTDIR" \
  --save-model missions_pyrenees_palisades_5sp5ev_grpbytype_sw4_directional2_k3_sims200000.zip \
  --progress-file-tag missions_pyrenees_palisades_5sp5ev_grpbytype_sw4_directional2_k3 \
  --chain-total-simulations 200000 \
  --simulations 200000 \
  --max-train-hours 140 \
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
