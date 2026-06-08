#!/usr/bin/env bash
#
# 3-day PPO training on Pyrenees_10p with variant_a arch.
# Periodic checkpointing every 100,000 timesteps guards against SLURM
# wall-limit termination: even if the run is killed at hour 72, the most
# recent <output-dir>/checkpoints/*.zip is preserved on disk and you can
# resume training with --load-model <that_zip> on a follow-up job.
#
# Sizing: 3,024,000 timesteps = 3x the 1M run that fit in 24h. Pyrenees_10p
# is slightly cheaper per step than Pyrenees_12p (-2 planes ~= -0.16 s/step
# at observed +0.08 s/plane). Expect comfortable fit in 72h; the checkpoint
# safety net handles any episode-length regressions as the policy learns.
#
#SBATCH -J pyr10p-vA-3d
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 72:00:00
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees10p_variant_a_3day_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_10p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 3024000 \
  --device cpu \
  --policy-arch variant_a \
  --checkpoint-interval 100000 \
  --output-dir "$OUTDIR" \
  --save-model pyrenees10p_variant_a_3day.zip \
  --progress-file-tag pyrenees10p_variant_a_3day \
  --vec-start-method auto \
  --gc-collect-on-reset
