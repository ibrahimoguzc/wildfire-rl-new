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
  --scenario "Pyrenees.json" \
  --num-envs 36 \
  --timesteps 518400 \
  --device cpu \
  --save-model "examples/wildfire/data/scenarios/outputs/pyrenees_3agent_3_13.zip" \
  --summary-output "examples/wildfire/data/scenarios/outputs/pyrenees_3agent_3_13_overall.xlsx" \
  --progress-file-tag "pyrenees_run_3agent_3_13" \
  --vec-start-method auto

