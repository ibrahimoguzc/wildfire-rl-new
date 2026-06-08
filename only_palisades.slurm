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
  --scenario "Palisades copy.json" \
  --num-envs 36 \
  --timesteps 1036800 \
  --device cpu \
  --save-model "examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip" \
  --summary-output "examples/wildfire/data/scenarios/outputs/onlypalisades_3agent_3_10_overall.xlsx" \
  --progress-file-tag "palisades_run_3agent_3_10" \
  --vec-start-method auto

