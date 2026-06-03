#!/usr/bin/env bash
#SBATCH -J pal-mixed
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
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

# === Thread pinning ===
# 36 forked envs on a 72-CPU allocation. Without these, each worker's BLAS
# (NumPy/PyTorch) spawns one thread PER core (36 x 72 ~= 2.6k threads) and they
# thrash the 72 cores -> single-digit fps. Cap each worker to 1 BLAS thread so
# the 36 envs step in parallel cleanly (still 2 cores/env of headroom).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_v2_${SLURM_JOB_ID}

/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades copy.json" \
  --num-envs 36 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_v2_switchign.zip \
  --progress-file-tag palisades_v2_switchign \
  --fire-detection-delay-minutes 60 \
  --vec-start-method auto \
  --state-fire-fronts 5 \
  --gc-collect-on-reset \
  --n-epochs 7 \
  --learning-rate 0.0004 \
  --state-space mixed \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --switch-ignition-2
