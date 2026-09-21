#!/usr/bin/env bash
#SBATCH -J conc-mw-150k
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

OUTDIR=examples/wildfire/data/scenarios/outputs/concurrent_multiweight_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_${SLURM_JOB_ID}

# CONCURRENT MULTIWEIGHT: the third multiweight arm, the analogue of the
# theater-level arm 5. --switch-scenarios assigns one scenario per worker in
# round-robin order and the worker keeps it for the whole run (ppo_runnerv2
# ~line 434); with --num-envs 128 and two scenarios that is a hard 64/64
# split - 64 workers only ever fly Palisades, 64 only ever Pyrenees, and
# every PPO batch mixes both maps. The runner prints the per-scenario worker
# counts at startup - check that line reads 64/64 before trusting the run.
#
# Both scenario JSONs carry the SAME per-type firefront cost weights
# (seaplanes 1.0 everywhere; eVTOLs 0.75 on vegetation/topography), so the
# weight treatment is uniform across the theater, matching the two dedicated
# multiweight runs (855200 Palisades, 769428 Pyrenees).
#
# Observation width is 69 (switch-scenarios forces the 3-way scenario
# one-hot), so checkpoints compare to the theater-level arms 3/4/5, NOT to
# the 66-feature single-map runs.
#
# 150,000 total simulations ACROSS both maps; the realised per-map episode
# split follows episode length, not allocation (arm 5 landed 44/56), and is
# recorded per scenario in the summary CSV - report it as-is.
#
# --chain-total-simulations puts the LR on the episode-driven schedule
# (5e-04 -> 0 across exactly 150000 episodes), identical to the sibling
# multiweight runs.
#
# RUNNER: ppo_runnerv3 - v2 plus the firefront-cost-weight log line and
# nothing else.
#
# Sizing: pal-mw measured ~2,200 sims/h (855200), pyr-mw ~1,850 (769428);
# the plain concurrent chain averaged ~2,200. Expect ~70-80 h for 150k;
# --max-train-hours 92 forces a clean save instead of a SIGKILL if the rate
# slips on a slow node.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv3 \
  --scenario "Pyrenees5sp5ev_multiweight.json" \
  --switch-scenarios "Palisades5sp5ev_multiweight.json" "Pyrenees5sp5ev_multiweight.json" \
  --output-dir "$OUTDIR" \
  --save-model concurrent_multiweight_5sp5ev_directional2_sims150000.zip \
  --progress-file-tag concurrent_multiweight_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k \
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
