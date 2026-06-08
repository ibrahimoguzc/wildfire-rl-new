#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH -t 12:00:00
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

python -u -m examples.wildfire.ppo_tester \
  --model-path examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip \
  --fixed-tactic-scenarios "Palisades_fixed_direct.json" "Palisades_fixed_dlr.json" "Palisades_fixed_indirect.json" "Palisades_fixed_mix.json"\
  --include-random-tactics \
  --scenarios "Palisades copy.json" \
  --runs-per-scenario 500 \
  --seed 1 \
  --device cpu \
  --num-workers 24 \
  --results-name palisades_results_3_22 \
  --summary-name palisades_summary_3_22

python -u -m examples.wildfire.ppo_tester \
  --model-path examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip \
  --fixed-tactic-scenarios "Salamis_fixed_direct.json" "Salamis_fixed_dlr.json" "Salamis_fixed_indirect.json" "Salamis_fixed_mix.json"\
  --include-random-tactics \
  --scenarios "Salamis.json" \
  --runs-per-scenario 500 \
  --seed 10000 \
  --device cpu \
  --num-workers 24 \
  --results-name salamis_results_3_22 \
  --summary-name salamis_summary_3_22


python -u -m examples.wildfire.ppo_tester \
  --model-path examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip \
  --fixed-tactic-scenarios "Pyrenees_fixed_direct.json" "Pyrenees_fixed_dlr.json" "Pyrenees_fixed_indirect.json" "Pyrenees_fixed_mix.json"\
  --include-random-tactics \
  --scenarios "Pyrenees.json" \
  --runs-per-scenario 500 \
  --seed 30000 \
  --device cpu \
  --num-workers 24 \
  --results-name pyrenees_results_3_22 \
  --summary-name pyrenees_summary_3_22
