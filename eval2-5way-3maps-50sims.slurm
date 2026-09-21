#!/usr/bin/env bash
#SBATCH -J eval2-5way-50
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 02:00:00
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
# TIME: 750 episodes x ~66 s / 64 workers ~= 15 min; 2 h wall is ample.
MODELS=(final_evaluation/models/arm1_palisades_only.zip
        final_evaluation/models/arm2_pyrenees_only.zip
        final_evaluation/models/arm5_concurrent.zip)

OUTDIR=examples/wildfire/data/scenarios/outputs/eval2_5way_3maps_50sims_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  echo "########## $SC ##########"
  python -u -m examples.wildfire.ppo_tester \
    --model-paths "${MODELS[@]}" \
    --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
    --include-random-tactics \
    --scenarios "$SC" \
    --runs-per-scenario 50 \
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

# Figures + stats via the parameterized eval_2 generators (suffix _50sims).
python -u scripts/plot_eval2_horizontal.py "$OUTDIR/results.csv" _50sims
python -u scripts/eval2_stat_tests.py "$OUTDIR/results.csv" \
  final_plots/eval_2/statistical_tests_50sims.md

echo "=== eval_2 done: $OUTDIR ==="
