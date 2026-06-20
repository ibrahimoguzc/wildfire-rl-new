#!/usr/bin/env bash
#SBATCH -J cont-pal-lr00075-
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

# Continue training the policy from job 557251.
# 557251 trained on the 3-aircraft "Palisades copy.json" (read once at startup).
# The scenario on disk is now 3 aircraft / correction_coefficient 0.40, so this
# is a faithful continuation under the same fleet. The runner re-reads the file
# fresh at startup, so keep it at the fleet/coeff you intend before submitting.
LOADMODEL=examples/wildfire/data/scenarios/outputs/palisades_lr00075_557251/palisades_lr00075.zip
OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_lr00075_cont557251_${SLURM_JOB_ID}

/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades copy.json" \
  --load-model "$LOADMODEL" \
  --num-envs 128 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_lr00075_cont.zip \
  --progress-file-tag palisades_lr00075_cont557251 \
  --vec-start-method fork \
  --state-fire-fronts 5 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --lr-decay-exponent 0.7 \
  --gamma 0.99 \
  --tactic-distribution individual \
  --controlled-agent-count 3 \
  --state-space old
