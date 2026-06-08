#!/usr/bin/env bash
#
#SBATCH -J rl-env
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
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
  --scenarios "Palisades copy.json" \
  --runs-per-scenario 100 \
  --seed 2000 \
  --device cpu \
  --num-workers 16 \
  --results-name palisades_results \
  --summary-name palisades_summary

python -u -m examples.wildfire.ppo_tester \
  --model-path examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip \
  --fixed-tactic-scenarios "Salamis_fixed_direct.json" "Salamis_fixed_dlr.json" "Salamis_fixed_indirect.json" "Salamis_fixed_mix.json"\
  --scenarios "Salamis.json" \
  --runs-per-scenario 100 \
  --seed 1000 \
  --device cpu \
  --num-workers 16 \
  --results-name salamis_results \
  --summary-name salamis_summary


python -u -m examples.wildfire.ppo_tester \
  --model-path examples/wildfire/data/scenarios/outputs/ppo_onlypalisades_3agent_3_10.zip \
  --fixed-tactic-scenarios "Pyrenees_fixed_direct.json" "Pyrenees_fixed_dlr.json" "Pyrenees_fixed_indirect.json" "Pyrenees_fixed_mix.json"\
  --scenarios "Pyrenees.json" \
  --runs-per-scenario 100 \
  --seed 3000 \
  --device cpu \
  --num-workers 16 \
  --results-name pyrenees_results \
  --summary-name pyrenees_summary
