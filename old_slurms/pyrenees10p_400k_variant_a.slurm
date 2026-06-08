#!/usr/bin/env bash
#
# PPO training on Pyrenees_10p with variant_a arch -- floor 400k timesteps.
# Target 432,000 timesteps; SB3 rounds up to a clean rollout boundary
# (36 envs * 144 n_steps = 5184 samples/rollout; 84 rollouts = 435,456).
#
# Wall sizing based on observed training rate, not profiling rate:
#   Jobs 507100 and 507102 (Pyrenees_12p, 36 envs, --policy-arch
#   variant_a / default) reached 181,440 timesteps in ~24h before the
#   wall ceiling -- ~2.13 timesteps/s. At that rate, 432k timesteps
#   takes ~56h. We budget 60h for safety. owl_normal_short qos caps at
#   24h and cannot be used here; running on normal_q (default qos).
#
# Periodic checkpointing every 50,400 timesteps (= 1400 * num_envs, so
# the actual num_timesteps in each filename is exactly 50,400/100,800/...
# rather than the off-by-32 you get with 50,000). Up to ~8-9 snapshots
# on disk so even if episode-length drift slows the run mid-flight, you
# still have a >=400k checkpoint preserved.
#
#SBATCH -J pyr10p-vA-400k
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 60:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=250G
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees10p_variant_a_400k_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_10p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 432000 \
  --device cpu \
  --policy-arch variant_a \
  --checkpoint-interval 50400 \
  --output-dir "$OUTDIR" \
  --save-model pyrenees10p_variant_a_400k.zip \
  --progress-file-tag pyrenees10p_variant_a_400k \
  --vec-start-method auto \
  --gc-collect-on-reset
