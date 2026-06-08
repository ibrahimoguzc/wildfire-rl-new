#!/usr/bin/env bash
#
# Full 1.008M-timestep PPO training on Pyrenees_12p with variant_a arch
# (shared 512->256 trunk + 128-wide pi/vf heads + ReLU). 36 parallel envs,
# --switch-ignition-1 for ignition randomisation, model saved at the end.
#
# Sizing: Pyrenees_12p profiled at ~2.9 s step_mean in the 12-env solo
# run (job 501353). With 36 envs amortising 3x more samples per parallel
# step, expected wall is ~24h for 1.008M timesteps. owl_normal_short qos
# gives that wall ceiling.
#
#SBATCH -J pyr12p-vA-1m
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

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees12p_variant_a_1m_$SLURM_JOB_ID

/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --scenario "Pyrenees_12p.json" \
  --switch-ignition-1 \
  --num-envs 36 \
  --timesteps 1008000 \
  --device cpu \
  --policy-arch variant_a \
  --output-dir "$OUTDIR" \
  --save-model pyrenees12p_variant_a_1m.zip \
  --progress-file-tag pyrenees12p_variant_a_1m \
  --vec-start-method auto \
  --gc-collect-on-reset
