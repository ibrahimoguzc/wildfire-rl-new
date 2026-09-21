#!/usr/bin/env bash
#SBATCH -J eval2-5way
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 03:00:00
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

# EVAL 2: 5 policies x 3 maps x 100 ignitions, seed-paired within map.
# Policies: the three completed plain 150k finals (arm1 Palisades-only,
# arm2 Pyrenees-only, arm5 concurrent - via the frozen symlinks under
# final_evaluation/models) + fixed heuristic + random tactics. Standard
# uniform-weight scenarios; same env flags as training/final eval.
# One tester invocation per map (post-OOM convention), 64 workers.
# TIME: 1,500 episodes x ~66 s / 64 workers ~= 30 min; 3 h wall is ample.
MODELS=(final_evaluation/models/arm1_palisades_only.zip
        final_evaluation/models/arm2_pyrenees_only.zip
        final_evaluation/models/arm5_concurrent.zip)

OUTDIR=examples/wildfire/data/scenarios/outputs/eval2_5way_3maps_100sims_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  echo "########## $SC ##########"
  python -u -m examples.wildfire.ppo_tester \
    --model-paths "${MODELS[@]}" \
    --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
    --include-random-tactics \
    --scenarios "$SC" \
    --runs-per-scenario 100 \
    --seed 1 --random-seeds \
    --device cpu --num-workers 64 \
    --state-space directional-2 --state-fire-fronts 3 \
    --water-set 2 --switch-ignition 4 \
    --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
    --output "$OUTDIR/results_${TAG}.csv" \
    --summary-output "$OUTDIR/summary_${TAG}.csv" \
    --plots-dir "$OUTDIR/plots_${TAG}"
done

# Merge and VERIFY the within-map seed pairing rather than assume it.
python - "$OUTDIR" <<'PYEOF'
import sys, glob, pandas as pd
out = sys.argv[1]
parts = sorted(glob.glob(f"{out}/results_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
for sc, g in df.groupby("scenario_name"):
    sets = {p: frozenset(x["seed"]) for p, x in g.groupby("policy_name")}
    assert len(set(sets.values())) == 1, f"seed sets differ on {sc}"
    print(f"  {sc}: {g['policy_name'].nunique()} policies x {len(next(iter(sets.values())))} shared seeds  OK")
df.to_csv(f"{out}/results.csv", index=False)
print(f"merged -> {len(df):,} rows")
PYEOF

# Four boxplots: one per mission + pooled, into final_plots/eval_2.
mkdir -p final_plots/eval_2
ORDER=(random_tactics fixed_Pyrenees5sp5ev_heuristic arm1_palisades_only arm2_pyrenees_only arm5_concurrent)
LABELS=("Random tactics" "Fixed tactics" "Palisades only" "Pyrenees only" "Theater level (concurrent)")
for SCEN in Palisades Pyrenees Salamis; do
  TAG=$(echo "$SCEN" | tr 'A-Z' 'a-z')
  python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
    --scenario "$SCEN" \
    --order "${ORDER[@]}" --labels "${LABELS[@]}" --color-by-type \
    --title "$SCEN · 100 paired simulations · switch-ignition-4, water set 2" \
    --out "final_plots/eval_2/eval_${TAG}.png"
done
python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
  --order "${ORDER[@]}" --labels "${LABELS[@]}" --color-by-type \
  --title "All three maps pooled · 3 x 100 paired simulations" \
  --note "Pooled over Palisades, Pyrenees and Salamis; seeds paired within each map" \
  --out "final_plots/eval_2/eval_pooled.png"

echo "=== eval_2 done: $OUTDIR ==="
