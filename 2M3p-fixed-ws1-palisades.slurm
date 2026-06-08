#!/usr/bin/env bash
#SBATCH -J 7ep-3p-ws1-fix
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=300G
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

# 36 forked envs on a 72-CPU allocation. Cap each worker's numerical
# libraries to one thread so the workers do not over-subscribe the node.
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_3p_lr00075_${SLURM_JOB_ID}

# 3 planes (from "Palisades copy.json"), water set 1, fixed (scenario)
# ignition, old state space. Fleet count comes from the scenario's
# agents_per_base; --controlled-agent-count must match it.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades copy.json" \
  --num-envs 128 \
  --timesteps 2016000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_3p_lr00075.zip \
  --progress-file-tag palisades_3p_lr00075 \
  --vec-start-method fork \
  --state-fire-fronts 5 \
  --gc-collect-on-reset \
  --n-epochs 7 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution individual \
  --controlled-agent-count 3 \
  --water-set 1 \
  --state-space old
