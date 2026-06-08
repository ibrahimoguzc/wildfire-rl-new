#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=300G
#SBATCH --qos=owl_normal_short
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
  --timesteps 208000 \
  --device cpu \
  --output-dir examples/wildfire/data/scenarios/outputs/4-26switch_scenario_fixed_ignition_1m \
  --save-model 4-26switch_scenario_fixed_ignition_1m.zip \
  --progress-file-tag 4-26switch_scenario_fixed_ignition_1m\
  --vec-start-method auto
