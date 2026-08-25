#!/usr/bin/env bash
#SBATCH -J eval-pyr5sp5ev
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ibrahimoguz@vt.edu

set -euo pipefail

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Each evaluation episode is a single-threaded rollout; cap the numerical
# libraries so 64 concurrent workers do not oversubscribe the 96-CPU node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# The 100,005-simulation checkpoint of the Pyrenees 5sp5ev chain (chunk 15 -
# the endpoint that closed the original 100k budget, before the 150k extension).
MODEL=examples/wildfire/data/scenarios/outputs/pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_chain/chunk15.zip

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_pyrenees_5sp5ev_3way_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# Three policies on identical fires:
#   1. random_tactics  - a fresh random tactic tuple per group at every
#      decision step (the tester samples from the env's own action space).
#   2. fixed_Pyrenees5sp5ev_heuristic - seaplanes defend urban/protection
#      locations (select_poi=vip), eVTOLs defend water resources
#      (select_poi=water); both follow_firefront + direct suppress. Derived
#      from Pyrenees5sp5ev_heuristic.json, which is Pyrenees5sp5ev.json with
#      only the two suppression_tactic blocks changed, so the fleet is
#      identical to what the model trained on.
#   3. chunk15 - the trained PPO policy, acting deterministically.
#
# --seed 1 --random-seeds draws 100 non-contiguous ignition seeds; every policy
# sees the same 100, so the comparison is paired rather than two independent
# samples. Reproducible: the same --seed redraws the same 100.
#
# The env flags must match training exactly or the observation/action shapes
# will not line up with the checkpoint: directional-2 state space, 3 fire
# fronts, water set 2, switch-ignition-4, 5+5 grouping over 10 aircraft, and
# the code-derived cell size.
python -u -m examples.wildfire.ppo_tester \
  --model-path "$MODEL" \
  --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
  --include-random-tactics \
  --scenarios Pyrenees5sp5ev.json \
  --runs-per-scenario 100 \
  --seed 1 \
  --random-seeds \
  --device cpu \
  --num-workers 64 \
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

python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
  --order random_tactics fixed_Pyrenees5sp5ev_heuristic chunk15 \
  --labels "Random tactics" "Fixed tactics" "Trained PPO (100k)" \
  --note "Fixed tactics = seaplanes defend urban, eVTOLs defend water; both follow firefront with direct suppression" \
  --title "Pyrenees 5sp+5ev · 100 paired simulations · switch-ignition-4, water set 2" \
  --out verification_plots/eval_pyr5sp5ev_3way_moe.png

echo "=== results: $OUTDIR ==="
