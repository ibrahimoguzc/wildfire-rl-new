#!/usr/bin/env bash
#SBATCH -J pyr5sp5ev-150k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_${SLURM_JOB_ID}

# Pyrenees, 5 example_seaplane_1 + 5 example_aircraft_1 (eVTOL) = 10 aircraft,
# grouped by type, 150,000 simulations, run as ONE job (no chunked chain) as
# requested. Supersedes job 717263 (the same run at 125,000), cancelled 23 min
# in at ~600 simulations; its output directory was removed.
#
# Settings are identical to the completed 200k chain (job 679348 -> 712753) so
# the two are directly comparable; the only differences are the budget and the
# fact that this one is a single process rather than 27 resumed chunks.
#
# On the budget: the 200k run learned entirely in its first ~100,000
# simulations and was statistically flat afterwards (100-150k vs 150-200k:
# +0.0014 +/- 0.0026, +0.5 sigma), finishing at 84.6% escapes. 150,000 is well
# past convergence, so expect ~84.5-85% escapes; the value of this run is a
# clean independent replicate of the 200k run at the ceiling, not a higher
# score. It also exactly matches the budget of the original (warm-restarted)
# 150k chain, which this run reproduces with one continuous LR schedule.
#
# --chain-total-simulations is set even though nothing is chained: it is what
# puts the LR on the episode-driven schedule so it decays 5e-04 -> 0 across
# exactly 125,000 simulations. With only --simulations the LR would decay
# against the timestep ceiling (125,000 x 144), which episodes never reach
# because they end when the fire escapes, leaving the run finishing at a
# nearly-undecayed LR.
#
# Sizing: the finished 200k chain did 200,002 simulations in 94.1 job-hours =
# 2,126 simulations/h for this exact fleet and config, so 150,000 is ~71 h
# (~6,800 CPU-hours). The rate climbs as the policy improves and episodes end
# sooner, so treat ~73 h as the pessimistic figure. The 96 h wall
# (owl_normal_base; owl_normal_short caps at 24 h) leaves comfortable margin,
# and --max-train-hours 92 makes the run save and exit on its own rather than
# being SIGKILLed if the rate slips badly.
#
# NOTE: a single job carries single-job risk - a crash loses everything since
# the last --checkpoint-interval save (every 252,000 timesteps, roughly every
# 14,800 simulations), where the chain would have resumed from its last chunk.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees5sp5ev.json" \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip \
  --progress-file-tag pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k \
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
