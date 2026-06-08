#!/usr/bin/env bash
#
# Solo profile of Salamis_12p with `enable_adaptive_time_step: false`.
# Compares directly against Salamis_12p in job 501353
# (adaptive=true, step_mean=3.59, reset_mean=21.33, free_burn=20.36 s)
# to quantify how much of Salamis's reset cost is the adaptive-timestep
# tax. Same training config (12 envs, 10368 timesteps, switch-ignition-1,
# gc-collect-on-reset) so deltas vs 501353/solo_salamis are apples-to-apples.
#
# Salamis_12p.json was edited to set enable_adaptive_time_step: false
# before this run.
#
#SBATCH -J salamis12p-noad
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 06:00:00
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

# Sanity check: confirm the flag is actually false in the JSON we are about
# to consume. The runner forks workers from this file; failing fast here
# avoids burning 5h on the wrong configuration.
python -c "
import json, sys
with open('examples/wildfire/data/scenarios/inputs/Salamis_12p.json') as f:
    d = json.load(f)
flag = d.get('enable_adaptive_time_step')
print(f'Salamis_12p.json enable_adaptive_time_step = {flag!r}')
if flag is not False:
    sys.exit(f'expected False, got {flag!r}')
"

NUM_ENVS=12
TIMESTEPS=10368  # matches job 501353/solo_salamis (6 rollouts at n_steps=144)

OUTDIR=examples/wildfire/data/scenarios/outputs/profile_salamis12p_noadaptive_$SLURM_JOB_ID
mkdir -p "$OUTDIR"

echo "============================================================"
echo ">>> SOLO PROFILE: Salamis_12p.json (adaptive_time_step=false)"
echo "============================================================"
/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Salamis_12p.json" \
  --switch-ignition-1 \
  --num-envs "$NUM_ENVS" \
  --timesteps "$TIMESTEPS" \
  --device cpu \
  --profile-timings \
  --gc-collect-on-reset \
  --output-dir "$OUTDIR" \
  --progress-file-tag "profile_salamis12p_noadaptive" \
  --vec-start-method auto

echo "============================================================"
echo ">>> DONE. Outputs under $OUTDIR"
echo "============================================================"
ls -la "$OUTDIR"

