#!/usr/bin/env bash
#SBATCH -J pal25k-pyr125k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_${SLURM_JOB_ID}

# SHORT PALISADES WARM-UP, then Pyrenees: 150000 simulations split 25000 /
# 125000 via --mission-weights 1 5 (weights are normalised, so 1/6 and 5/6 of
# the budget; verified against _split_budget -> [25000, 125000] exactly).
# Sequential phases: the whole fleet finishes Palisades before any worker sees
# Pyrenees. Confirm at startup with:
#   Mission phase budgets (150000 simulations, weights 1/5), run in this order:
#     Palisades5sp5ev.json -> 25000 simulations then Pyrenees5sp5ev.json -> 125000
#
# WHY THIS SPLIT. The 50/50 runs (679425 decaying LR, 684399 constant LR) spent
# half the budget on a map that saturates at ~3% escapes, and the held-out
# 1000-ignition evaluation showed the cost: Pyrenees-only scored +0.024 MoE on
# Pyrenees against -0.037 for Palisades->Pyrenees, a 6.2 sigma gap on equal
# total budget. Yet Palisades itself is nearly free to learn - the 50/50 runs
# were already at ~5% escapes by 25000 simulations and only crawled from 5% to
# 3% over the second 25000. So 25000 should buy essentially all of the
# Palisades competence for a fifth of the price, leaving 125000 for the map
# that actually has headroom.
#
# The LR side matters as much as the budget side. With the default exponent
# 0.70 over 150000 simulations:
#     sim      0  LR 5.000e-04   Palisades starts
#     sim 25,000  LR 4.401e-04   switch to Pyrenees  (88% of full step size)
#     sim 150,000 LR ~0
# Compare the 50/50 100k runs, where Pyrenees inherited only 3.077e-04 (62%).
# A short warm-up therefore removes most of the LR handicap that confounded the
# earlier curriculum results, without needing --lr-decay-exponent 0.
#
# This is the third point on the Palisades-share axis:
#     0%   of budget on Palisades  -> job 717319 (Pyrenees only, 150k)
#     17%  of budget on Palisades  -> THIS RUN
#     50%  of budget on Palisades  -> 679425 / 684399 (at 100k)
# 717319 is the matched control: same 150000 total, same fleet, same schedule
# shape, same single-job form - it differs only in having no Palisades phase.
#
# 10 aircraft on both maps: 5 example_seaplane_1 + 5 example_aircraft_1,
# seaplanes listed first in both scenario JSONs, so --group-sizes 5 5 means the
# same thing in both phases. --switch-ignition-4, water set 2, directional-2
# (K=3), --cell-size code: identical to every other run in the study.
#
# NOTE: multi-mission runs add the 3 scenario one-hot features, so this policy
# has 69 observation features against 66 for the single-map runs. Its
# checkpoints are not interchangeable with 717319's.
#
# Sizing: 25000 Palisades at the measured 2430 sims/h is ~10.3 h; 125000
# Pyrenees at 2126 sims/h (from the completed 200k chain) is ~58.8 h; ~69 h
# total, ~6,600 CPU-hours. The 96 h wall leaves margin and --max-train-hours 92
# makes the run save and exit cleanly if the rate slips.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --missions "Palisades5sp5ev.json" "Pyrenees5sp5ev.json" \
  --mission-weights 1 5 \
  --output-dir "$OUTDIR" \
  --save-model missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip \
  --progress-file-tag missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3 \
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
