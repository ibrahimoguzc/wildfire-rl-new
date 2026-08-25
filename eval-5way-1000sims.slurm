#!/usr/bin/env bash
#SBATCH -J eval-5way-1000
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 06:00:00
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
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_5way_1000_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# Five policies x two maps x 1000 held-out ignitions = 10,000 episodes.
#
# All five in ONE invocation so a single --seed 1 --random-seeds draw supplies
# the same 1000 ignition seeds to every policy on a given map. (The earlier
# 100-sim result needed two jobs merged by hand; this avoids that.)
#
# Models are symlinks under eval_models/ purely so policy_name in the CSV reads
# as ppo_pyrenees_only / ppo_palisades_only / ppo_pal_then_pyr instead of
# chunk15 / latest / missions_..._sims100000.
#   ppo_pyrenees_only   Pyrenees only, 100,005 sims (66 obs features)
#   ppo_palisades_only  Palisades only,  93,800 sims (66) - short of 100k, its
#                       chain stopped on the MAX_CHUNKS bug; curve was flat
#                       from ~60k so this is a converged policy regardless
#   ppo_pal_then_pyr    job 679425, 50k Palisades then 50k Pyrenees (69 obs:
#                       carries the 3-way scenario one-hot; ppo_tester rebuilds
#                       the env with include_scenario_features for it)
# plus the fixed-tactic baseline (seaplanes->urban, eVTOLs->water,
# follow_firefront + direct; the derived action is map-independent) and the
# random-tactics baseline.
#
# Sizing: job 710905 did 800 episodes in 15.6 min on 64 workers (~75 s per
# episode per worker). 10,000 episodes on 96 workers is ~2.2 h; the 6 h wall
# leaves room for the longer Palisades episodes the baselines produce.
python -u -m examples.wildfire.ppo_tester \
  --model-paths eval_models/ppo_pyrenees_only.zip \
                eval_models/ppo_palisades_only.zip \
                eval_models/ppo_pal_then_pyr.zip \
  --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
  --include-random-tactics \
  --scenarios Pyrenees5sp5ev.json Palisades5sp5ev.json \
  --runs-per-scenario 1000 \
  --seed 1 \
  --random-seeds \
  --device cpu \
  --num-workers 96 \
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

for SC in Pyrenees5sp5ev.json Palisades5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
    --scenario "$SC" --color-by-type \
    --order random_tactics fixed_Pyrenees5sp5ev_heuristic \
            ppo_palisades_only ppo_pyrenees_only ppo_pal_then_pyr \
    --labels "Random tactics" "Fixed tactics" "PPO Palisades-only" \
             "PPO Pyrenees-only" "PPO Pal→Pyr" \
    --note "Grey = non-learned baseline · coloured = trained policy · 5sp+5ev fleet on every arm" \
    --title "${SC%%5sp5ev.json} · 1000 paired simulations · switch-ignition-4, water set 2" \
    --out "verification_plots/eval_5way_1000_${TAG}.png"
done

echo "=== results: $OUTDIR ==="
