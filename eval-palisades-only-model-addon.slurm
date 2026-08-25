#!/usr/bin/env bash
#SBATCH -J eval-palonly
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail
mkdir -p logs
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Only the policy job 710905 was missing: the Palisades-ONLY 5sp5ev model
# (93,800 simulations - that chain stopped early on the MAX_CHUNKS bug, so it
# is not a full 100k like its Pyrenees sibling).
#
# Same --seed 1 --random-seeds and the same scenarios, so ppo_tester redraws
# the identical 100 ignition seeds per map and these rows merge straight into
# 710905's results.csv as a fifth policy, still fully paired.
PAL_ONLY=examples/wildfire/data/scenarios/outputs/palisades_5sp5ev_grpbytype_sw4_directional2_k3_chain/latest.zip

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_palonly_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

python -u -m examples.wildfire.ppo_tester \
  --model-path "$PAL_ONLY" \
  --scenarios Pyrenees5sp5ev.json Palisades5sp5ev.json \
  --runs-per-scenario 100 \
  --seed 1 \
  --random-seeds \
  --device cpu \
  --num-workers 32 \
  --state-space directional-2 \
  --state-fire-fronts 3 \
  --water-set 2 \
  --switch-ignition 4 \
  --group-sizes 5 5 \
  --controlled-agent-count 10 \
  --cell-size code \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"

echo "=== results: $OUTDIR ==="
