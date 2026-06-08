#!/usr/bin/env bash
#
#SBATCH -J rl-test
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=200G
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

python -u -m examples.wildfire.ppo_runner \
  --switch-scenario \
  --num-envs 36 \
  --timesteps 15000 \
  --profile-timings \
  --device cpu \
  --output-dir examples/wildfire/data/scenarios/outputs/profile_timings \
  --progress-file-tag profile_timings \
  --vec-start-method auto
