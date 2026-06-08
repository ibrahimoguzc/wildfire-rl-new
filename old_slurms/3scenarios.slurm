#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 72:00:00
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
  --num-envs 48 \
  --timesteps 1036800 \
  --device cpu \
  --output-dir examples/wildfire/data/scenarios/outputs/3scenarios_3-30 \
  --save-model 3scenarios_3_30.zip \
  --progress-file-tag 3scenarios-results-3-30\
  --vec-start-method auto
