#!/usr/bin/env bash
#
# PPO training on Pyrenees_9p with variant_a arch + tuned hyperparams to
# break the ~150k plateau seen in job 512195 (high approx_kl=0.037,
# clip_fraction=0.343, entropy stalling, explained_variance plateaued).
#
# Hyperparameter changes vs prior 9p run (Conservative-minus-gamma):
#   --learning-rate 2e-4       (was 3e-4)   smaller actor steps
#   --target-kl 0.02                          KL gate: epochs early-stop
#                                              when approx_kl > 0.03
#   --n-epochs 5               (was 7)        fewer passes per rollout
#   --lr-decay-exponent 1.0    (was 0.65)     linear decay, keeps the
#                                              actor moving past iter 30
#   --gamma 0.99 (default)                    kept: Pyrenees episodes are
#                                              short (6-14 steps), the
#                                              terminal MoE signal must
#                                              reach early steps intact
#
# Target 207,360 timesteps = 40 rollouts * 5184 samples/rollout (clean
# SB3 rollout boundary above the 200k floor). At observed Pyrenees_12p
# rate (~2.13 ts/s, jobs 507100/507102), 9p extrapolates to ~2.30 ts/s.
# n_epochs=5 vs 7 saves ~7% on optimization, so effective ~2.46 ts/s.
# 207,360 / 2.46 ~= 84,000 s ~= 23.3 h, fits owl_normal_short 24h cap
# with ~40 min buffer.
#
# Checkpoint every 50,400 timesteps (= 1400 * num_envs) gives 4 saves
# before the 200k floor: 50.4k / 100.8k / 151.2k / 201.6k. So even if
# the wall hits early due to episode-length drift, a >=150k snapshot
# is preserved on disk.
#
#SBATCH -J pyr9p-vA-200k-kl
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees9p_variant_a_200k_kltuned_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_9p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 180000 \
  --device cpu \
  --policy-arch variant_a \
  --learning-rate 2e-4 \
  --target-kl 0.02 \
  --n-epochs 5 \
  --lr-decay-exponent 1.0 \
  --checkpoint-interval 50400 \
  --output-dir "$OUTDIR" \
  --save-model pyrenees9p_variant_a_200k_kltuned.zip \
  --progress-file-tag pyrenees9p_variant_a_200k_kltuned \
  --vec-start-method auto \
  --gc-collect-on-reset
