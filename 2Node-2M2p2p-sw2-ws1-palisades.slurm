#!/usr/bin/env bash
#SBATCH -J 2p2p-sw2-ws1
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
##SBATCH --mem=300G
#SBATCH --qos=owl_normal_short

set -euo pipefail

mkdir -p logs
#MPI-to get to nodes on sync
#distributed runs 
#
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

OUTDIR=examples/wildfire/data/scenarios/outputs/palisades_2p2p_lr00075_${SLURM_JOB_ID}

# 4 planes grouped into 2 groups of 2 (one tactic decision per group ->
# ceil(4/2)=2 decisions), water set 1, switch-ignition-2, old state space.
/usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
  --scenario "Palisades.json" \
  --num-envs 128 \ 
  --timesteps 2016000 \
  --device cpu \
  --output-dir "$OUTDIR" \
  --save-model palisades_2p2p_lr00075.zip \
  --progress-file-tag palisades_2p2p_lr00075 \
  --vec-start-method fork \
  --state-fire-fronts 5 \
  --gc-collect-on-reset \
  --n-epochs 5 \
  --learning-rate 0.0005 \
  --gamma 0.99 \
  --tactic-distribution group \
  --aircraft-group-size 2 \
  --controlled-agent-count 4 \
  --water-set 1 \
  --switch-ignition-2 \
  --state-space old
