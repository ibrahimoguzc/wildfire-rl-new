#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -t 23:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=100G
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

python -u -m examples.wildfire.test_fixed_tactics_from_seed_file \
  --input examples/wildfire/data/scenarios/outputs/CASData/Figure4.xlsx \
  --seed-column sim_seed \
  --scenario-column scenario_path \
  --tactic-source-scenario "Palisades_fixed_direct.json" \
  --num-workers 24 \
  --checkpoint-every 1000 \
  --output examples/wildfire/data/scenarios/outputs/CASData/figure4_fixed_direct_results.csv \
  --summary-output examples/wildfire/data/scenarios/outputs/CASData/figure4_fixed_direct_summary.csv
