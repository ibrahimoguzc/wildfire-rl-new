#!/usr/bin/env bash
#
# Full 1.008M-timestep PPO training on Pyrenees_12p with the DEFAULT arch
# (independent pi=[512,256,128] / vf=[512,256,128] Tanh stacks, ~420k params).
# Companion to pyrenees12p_1m_variant_a.slurm; same flags otherwise so the
# two runs differ only in --policy-arch.
#
# Note: planning to also drop Pyrenees_12p.json `wind_run` from 350 -> 340
# before submitting this. That changes the fire regime (slightly less
# spread, longer episodes expected), so do NOT compare numbers from this
# job against the variant_a job 507097-style runs that used wind_run=350.
#
#SBATCH -J pyr12p-def-1m
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=250G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees12p_default_1m_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_12p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 1008000 \
  --device cpu \
  --policy-arch default \
  --output-dir "$OUTDIR" \
  --save-model pyrenees12p_default_1m.zip \
  --progress-file-tag pyrenees12p_default_1m \
  --vec-start-method auto \
  --gc-collect-on-reset
