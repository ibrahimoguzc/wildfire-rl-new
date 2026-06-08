#!/usr/bin/env bash
#
# Test PPO training cost on a 12-plane fleet, applied uniformly to all
# three scenarios. Lets us isolate "agent count" as a variable: the
# previous job 500674 had Palisades=3 / Pyrenees=14 / Salamis=3, so the
# step-time gap conflated agent count with map-driven cost. Here all three
# carry 12 indirect-tactic seaplanes from
#   examples/wildfire/data/scenarios/inputs/{Palisades,Pyrenees,Salamis}_12p.json
#
# Phases:
#   1) Solo PPO run per scenario with --profile-timings (12 envs, 6 rollouts).
#   2) Switch-scenario run round-robin across all three (now possible
#      because all three have matching agent counts). One process, same
#      OS/cache state, direct apples-to-apples comparison.
#
# Compare against profile_500674 to attribute the imbalance to
# (a) per-agent ABM cost vs (b) map/terrain-driven CA cost vs
# (c) reset/free-burn cost (water polygons + adaptive timestep).
#
#SBATCH -J profile-12p
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 12:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=160G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

NUM_ENVS=12
# 6 rollouts × 144 n_steps × 12 envs. Pyrenees solo at 14 planes took
# 2h30m for this size; Pyrenees_12p should be a bit faster (3 fewer
# planes × ABM cost), Salamis_12p will be slower than 3-plane Salamis.
SOLO_TIMESTEPS=10368
# Switch phase: 4 rollouts is enough for stable per-scenario means.
SWITCH_TIMESTEPS=6912

OUTDIR_BASE=examples/wildfire/data/scenarios/outputs/profile_12p_$SLURM_JOB_ID
mkdir -p "$OUTDIR_BASE"

run_solo () {
  local scenario_json="$1"
  local tag="$2"
  echo "============================================================"
  echo ">>> SOLO PROFILE (12-plane): $scenario_json"
  echo "============================================================"
  /usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
    --scenario "$scenario_json" \
    --switch-ignition-1 \
    --num-envs "$NUM_ENVS" \
    --timesteps "$SOLO_TIMESTEPS" \
    --device cpu \
    --profile-timings \
    --gc-collect-on-reset \
    --output-dir "$OUTDIR_BASE/solo_${tag}" \
    --progress-file-tag "profile_12p_${tag}" \
    --vec-start-method auto
}

# Phase 1: solo runs.
run_solo "Palisades_12p.json" "palisades"
run_solo "Pyrenees_12p.json"  "pyrenees"
run_solo "Salamis_12p.json"   "salamis"

# Phase 2: switch-scenario round-robin (validates the equal-agent-count
# assumption enforced by _validate_switch_scenario_agent_counts).
echo "============================================================"
echo ">>> SWITCH-SCENARIO PROFILE (12-plane round-robin)"
echo "============================================================"
/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --switch-scenario \
  --switch-scenarios "Palisades_12p.json" "Pyrenees_12p.json" "Salamis_12p.json" \
  --switch-ignition-1 \
  --num-envs "$NUM_ENVS" \
  --timesteps "$SWITCH_TIMESTEPS" \
  --device cpu \
  --profile-timings \
  --gc-collect-on-reset \
  --output-dir "$OUTDIR_BASE/switch_all" \
  --progress-file-tag "profile_12p_switch" \
  --vec-start-method auto

echo "============================================================"
echo ">>> DONE. Outputs under $OUTDIR_BASE"
echo "============================================================"
ls -la "$OUTDIR_BASE"
