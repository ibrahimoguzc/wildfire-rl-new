#!/usr/bin/env bash
#SBATCH -J dir-k2-smoke
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short

set -euo pipefail
mkdir -p logs

module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Quick smoke / timing check for the "directional" state (K=2): ~2 rollouts.
python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades.json" \
  --num-envs 16 \
  --timesteps 4608 \
  --device cpu \
  --vec-start-method fork \
  --state-space directional \
  --state-fire-fronts 2 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --controlled-agent-count 4 \
  --water-set 1 \
  --switch-ignition-2
