#!/usr/bin/env bash
#SBATCH -J pal5sp5ev-150k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_5sp5ev_grpbytype_sw4_directional2_k3_150k_${SLURM_JOB_ID}

# PALISADES ONLY, 150000 simulations, single job. The mirror of 717319
# (Pyrenees only, 150000) and the missing single-map endpoint of the study.
#
# WHY IT IS WORTH RUNNING despite Palisades being saturated: every existing
# Palisades-only number comes from one chain that stopped at 93,800 on the
# MAX_CHUNKS bug, so it is the only arm not at a matched 150000 budget. This
# run makes the single-map pair symmetric (717319 Pyrenees 150k / this
# Palisades 150k), which is what the transfer matrix needs: the held-out
# evaluation compares "trained on X, evaluated on Y" and an unequal budget on
# one axis weakens every cell in that row.
#
# WHAT TO EXPECT: not much. The 93,800-simulation chain was flat from ~60,000
# and finished at +0.979 MoE / 1% escapes on held-out ignitions, already the
# best Palisades score of any policy. The extra 56,000 simulations here should
# change little - the value is a matched budget and a second, cleanly
# single-job Palisades policy, not a better one. If it does improve materially,
# that itself is worth knowing, since it would mean the earlier chain stopped
# before convergence rather than at it.
#
# --chain-total-simulations is set even though nothing is chained: it puts the
# LR on the episode-driven schedule so it decays 5e-04 -> 0 across exactly
# 150000 simulations, matching 717319. With only --simulations the LR would
# decay against the timestep ceiling that episodes never reach.
#
# 10 aircraft: 5 example_seaplane_1 + 5 example_aircraft_1, seaplanes first, so
# --group-sizes 5 5 gives one tactic decision per type. --switch-ignition-4,
# water set 2, directional-2 (K=3), --cell-size code: identical to 717319, so
# the two single-map runs differ only in the map. Observation width is 66 (no
# scenario one-hot), so its checkpoints are interchangeable with 717319's and
# NOT with the multi-mission runs'.
#
# Sizing: Palisades ran at 2430 simulations/h for this fleet (job 677151), so
# 150000 is ~62 h (~5,900 CPU-hours). Palisades episodes shorten as containment
# improves, so treat that as the pessimistic figure. The 96 h wall leaves
# margin and --max-train-hours 92 forces a clean save if the rate slips.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades5sp5ev.json" \
  --output-dir "$OUTDIR" \
  --save-model palisades_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip \
  --progress-file-tag palisades_5sp5ev_grpbytype_sw4_directional2_k3_150k \
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
