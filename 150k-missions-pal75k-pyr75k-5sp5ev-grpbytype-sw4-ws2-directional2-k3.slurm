#!/usr/bin/env bash
#SBATCH -J pal75k-pyr75k
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

module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/missions_pal75k_pyr75k_5sp5ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# EVEN SEQUENTIAL SPLIT: 150000 simulations as 75000 Palisades then 75000
# Pyrenees. --mission-weights 1 1 (equal shares; verified _split_budget ->
# [75000, 75000]). Sequential phases, so the whole fleet finishes Palisades
# before any worker sees Pyrenees. Confirm at startup with:
#   Mission phase budgets (150000 simulations, weights 1/1), run in this order:
#     Palisades5sp5ev.json -> 75000 simulations then Pyrenees5sp5ev.json -> 75000
#
# This fills the last gap in the Palisades-share axis, all at 150000 total,
# same fleet, same conditions, all single jobs except the concurrent one:
#     0%   Palisades  717319  Pyrenees only                    -> 84.2% escapes
#     17%  Palisades  717320  25k Palisades -> 125k Pyrenees    (running)
#     50%  Palisades  THIS    75k Palisades -> 75k Pyrenees
#     ~44% Palisades  717323  concurrent 64/64 workers          (running)
# The 50/50 point previously existed only at 100k (679425 decaying LR, 684399
# constant LR), so it could not be compared like-for-like against the 150k
# arms. This run makes the whole axis matched-budget.
#
# It is also the direct sequential control for the concurrent run 717323, which
# is splitting its 150000 roughly 44/56 Palisades/Pyrenees in practice. Same
# budget, same two maps, near-enough the same per-map exposure - the difference
# is purely whether the maps are interleaved in the gradient or separated in
# time. That is the cleanest sequential-vs-concurrent test in the study.
#
# EXPECTED WEAKNESS, stated up front: with the default exponent 0.70 over
# 150000, Pyrenees inherits LR 3.078e-04 at the switch (62% of full step size)
# and anneals to zero from there, exactly the handicap that made the 100k 50/50
# runs look worse than they were. 717320's short warm-up hands Pyrenees
# 4.401e-04 (88%) by comparison. So if this arm underperforms, the LR is a
# candidate explanation and 684399's constant-LR result is the reference for
# how much of the gap that accounts for. Keeping the default here is
# deliberate: it makes this run the 150k twin of 679425 rather than a new
# combination of two changes at once.
#
# 10 aircraft on both maps: 5 example_seaplane_1 + 5 example_aircraft_1,
# seaplanes first in both scenario JSONs so --group-sizes 5 5 means the same
# thing in both phases. --switch-ignition-4, water set 2, directional-2 (K=3),
# --cell-size code: identical to every other run in the study.
#
# NOTE: multi-mission runs carry the 3 scenario one-hot features, so this
# policy has 69 observation features against 66 for the single-map runs; its
# checkpoints are not interchangeable with 717319's.
#
# Sizing: 75000 Palisades at 2430 sims/h is ~31 h; 75000 Pyrenees at 2126
# sims/h (measured on the completed 200k chain) is ~35 h; ~66 h total,
# ~6,400 CPU-hours. The 96 h wall leaves margin and --max-train-hours 92 makes
# the run save and exit cleanly rather than being SIGKILLed if the rate slips.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --missions "Palisades5sp5ev.json" "Pyrenees5sp5ev.json" \
  --mission-weights 1 1 \
  --output-dir "$OUTDIR" \
  --save-model missions_pal75k_pyr75k_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip \
  --progress-file-tag missions_pal75k_pyr75k_5sp5ev_grpbytype_sw4_directional2_k3 \
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
