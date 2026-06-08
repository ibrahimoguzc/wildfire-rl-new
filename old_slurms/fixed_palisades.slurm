#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
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

python -u -m examples.wildfire.test_fixed_tactics_from_seed_file \
  --input examples/wildfire/data/scenarios/outputs/onlypalisades_3agent_3_10_overall.xlsx \
  --seed-column sim_seed \
  --scenario-column scenario \
  --num-workers 12 \
  --checkpoint-every 500 \
  --output examples/wildfire/data/scenarios/outputs/my_fixed_tactics_results.csv \
  --summary-output examples/wildfire/data/scenarios/outputs/my_fixed_tactics_summary.csv\
  --palisades-scenario "Palisades_fixed.json" \
  --pyrenees-scenario "Pyrenees.json" \
  --salamis-scenario "Salamis.json" 

