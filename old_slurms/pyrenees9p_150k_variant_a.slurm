#!/usr/bin/env bash
#
# PPO training on Pyrenees_9p with variant_a arch -- floor 150k timesteps.
# Target 155,520 (= 30 rollouts * 5184 samples/rollout), so SB3 finishes
# on a clean PPO boundary safely above the 150k floor.
#
# Wall sizing from observed Pyrenees_12p training rate (~2.13 ts/s at
# 36 envs, jobs 507100/507102). 9 planes = 3 fewer than 12p; at ~0.08s
# per plane saved (from the per-plane ABM step cost in our 12p vs 14p
# profiling diff), expect ~9% higher throughput -> ~2.32 ts/s.
# 155,520 / 2.32 ~= 67,000 s ~= 18.6 h. 24h wall on owl_normal_short
# gives ~5h buffer for episode-length drift as the policy learns.
#
# Checkpointing every 25,200 timesteps (= 700 * num_envs) gives 6 saves
# before the 150k floor: 25.2k / 50.4k / 75.6k / 100.8k / 126k / 151.2k.
#
#SBATCH -J pyr9p-vA-150k
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=250G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees9p_variant_a_150k_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_9p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 155520 \
  --device cpu \
  --policy-arch variant_a \
  --output-dir "$OUTDIR" \
  --save-model pyrenees9p_variant_a_150k.zip \
  --progress-file-tag pyrenees9p_variant_a_150k \
  --vec-start-method auto \
  --gc-collect-on-reset
