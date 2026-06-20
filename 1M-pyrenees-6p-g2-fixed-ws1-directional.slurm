#!/usr/bin/env bash
#SBATCH -J pyr6-dir-fix
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# 128 forked envs on a 96-CPU allocation. Cap each worker's numerical
# libraries to one thread so the workers do not over-subscribe the node.
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_6p_g2_directional_k2_${SLURM_JOB_ID}

# 6 planes in 3 groups of 2, water set 1, fixed (scenario) ignition,
# directional state space (K=2). Pyrenees.json defines 14 aircraft, so 6
# controlled aircraft is valid.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees.json" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_6p_g2_directional_k2.zip \
  --progress-file-tag pyrenees_6p_g2_directional_k2 \
  --vec-start-method fork \
  --state-fire-fronts 2 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --controlled-agent-count 6 \
  --water-set 1 \
  --state-space directional
