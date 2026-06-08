#!/usr/bin/env bash
#SBATCH -J pyr-mix-ws1
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
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
# 40 forked envs on an 80-CPU allocation. Without these, each worker's BLAS
# (NumPy/PyTorch) spawns one thread PER core and they thrash the cores ->
# single-digit fps. Cap each worker to 1 BLAS thread so the 40 envs step in
# parallel cleanly (2 cores/env of headroom).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/pyrenees_v2_ws1_${SLURM_JOB_ID}

# Smallest water set for Pyrenees: set1 = bodies the current fleet can scoop
# (19 OSM bodies), vs full=140 / set2=116. Requires the pre-generated pkl:
#   Pyrenees_2000x1996_5m_water_sources_set1.pkl
# (generate via: python -m examples.wildfire.generate_water_sets
#                  --scenario "Pyrenees.json")
# This is the high-payoff case: Pyrenees' 140-body water-feasibility check is
# the bulk of its ~6s reset cost; set1 cuts it to 19 bodies.
#
# 12 controlled planes in 3 groups of 4 (ceil(12/4)=3). Pyrenees.json has 14.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Pyrenees.json" \
  --num-envs 40 \
  --timesteps 1008000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model pyrenees_v2_switchign_ws1.zip \
  --progress-file-tag pyrenees_v2_switchign_ws1 \
  --fire-detection-delay-minutes 30 \
  --vec-start-method auto \
  --state-fire-fronts 5 \
  --gc-collect-on-reset \
  --n-epochs 7 \
  --learning-rate 0.0004 \
  --gamma 0.99 \
  --state-space mixed \
  --tactic-distribution group \
  --aircraft-group-size 4 \
  --controlled-agent-count 12 \
  --water-set 1 \
  --switch-ignition-2
