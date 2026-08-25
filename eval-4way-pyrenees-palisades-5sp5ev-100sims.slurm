#!/usr/bin/env bash
#SBATCH -J eval-4way-5sp5ev
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

module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Pyrenees-only policy: chunk 15 of the original pyrenees_5sp5ev chain, the
# 100,005-simulation endpoint of a *completed* 100k LR schedule (fully annealed).
PYR_ONLY=examples/wildfire/data/scenarios/outputs/pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_chain/chunk15.zip

# Multi-mission policy: job 679425's final model, 50k Palisades then 50k Pyrenees.
MULTI=examples/wildfire/data/scenarios/outputs/missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_679425/missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims100000.zip

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_4way_5sp5ev_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# Four policies on BOTH maps, 100 paired simulations each (800 episodes total):
#   random_tactics  - fresh random tactic per group at every decision step
#   fixed_...       - seaplanes defend urban (vip), eVTOLs defend water, both
#                     follow_firefront + direct. The fixed action is derived
#                     from the scenario's tactic blocks and is map-independent
#                     (it resolves to [6, 2] whichever map is being flown), so
#                     one heuristic file serves both scenarios.
#   chunk15         - trained on Pyrenees only, 100k simulations
#   missions_...    - trained Palisades->Pyrenees, 100k simulations (679425)
#
# Observation widths differ: the Pyrenees-only model has 66 features, the
# multi-mission model 69 (it carries the 3-way scenario one-hot). ppo_tester
# detects env_obs + len(SCENARIO_FEATURES) == model_obs and rebuilds the env
# with include_scenario_features=True for that model. The one-hot is resolved
# from the scenario FILENAME (_resolve_scenario_one_hot: "palisades" -> 1,0,0,
# "pyrenees" -> 0,1,0), so it means the same thing here as it did in training.
#
# --seed 1 --random-seeds draws 100 non-contiguous ignition seeds per scenario;
# every policy sees the same 100 on a given map, so all comparisons are paired.
python -u -m examples.wildfire.ppo_tester \
  --model-paths "$PYR_ONLY" "$MULTI" \
  --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
  --include-random-tactics \
  --scenarios Pyrenees5sp5ev.json Palisades5sp5ev.json \
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

MULTI_NAME=missions_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims100000
for SC in Pyrenees5sp5ev.json Palisades5sp5ev.json; do
  TAG=$(echo "$SC" | sed 's/5sp5ev.json//' | tr 'A-Z' 'a-z')
  python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
    --scenario "$SC" \
    --order random_tactics fixed_Pyrenees5sp5ev_heuristic chunk15 "$MULTI_NAME" \
    --labels "Random tactics" "Fixed tactics" "PPO: Pyrenees-only 100k" "PPO: Pal→Pyr 100k" \
    --note "Fixed tactics = seaplanes defend urban, eVTOLs defend water; both follow firefront with direct suppression" \
    --title "${SC%%5sp5ev.json} · 5sp+5ev · 100 paired simulations · switch-ignition-4, water set 2" \
    --out "verification_plots/eval_4way_${TAG}.png"
done

echo "=== results: $OUTDIR ==="
