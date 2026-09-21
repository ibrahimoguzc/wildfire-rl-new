#!/usr/bin/env bash
#SBATCH -J final-eval-8way
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 06:00:00
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

# THE FINAL EVALUATION. 8 policies x 3 maps x 250 held-out ignitions = 6,000
# episodes, everything into final_evaluation/.
#
# Policies (models are symlinks under final_evaluation/models, all verified to
# load before submission):
#   arm1  Palisades only        150k   (66 obs features)
#   arm2  Pyrenees only         150k   (66)
#   arm3  even split 75k/75k    150k   (69 - carries the scenario one-hot)
#   arm4  Pal 25k -> Pyr 125k   150k   (69)
#   arm5  concurrent 64/64      150k   (69; achieved split 44/56, not even)
#   arm6  Palisades MULTIWEIGHT  25k   (66; per-type cost weights, eVTOLs 0.5
#         on priority/vegetation/topography - note the much smaller budget,
#         625k timesteps vs ~2.5-3.2M for the others)
#   fixed tactics  (seaplanes->urban VIP, eVTOLs->water; follow_firefront +
#                   direct; the derived action [6,2] is map-independent)
#   random tactics (fresh random tactic per group each decision step)
#
# This is the first evaluation to use the TRUE 150k finals for arms 1-5 - every
# earlier run used ~100k stand-ins, two of them mid-LR-schedule.
#
# SEED DISCIPLINE: within a map, one --seed 1 --random-seeds draw supplies the
# identical 250 ignitions to all 8 policies, so every within-map comparison is
# fully paired (verified post-hoc by the merge step below, which fails loudly
# if any policy's seed set differs).
#
# ONE INVOCATION PER MAP, 64 workers - not one big run at 96. Job 753124 ran
# all maps in one process at 96 workers and was OOM-killed at 78% (MaxRSS
# 821 GB / 761 GB node): env memory grows during long episodes as fire grids
# expand. Per-map processes release memory between maps and cap a failure's
# cost at one map; 64 workers held ~547 GB on the successful rerun 754407.
#
# "Learning should stop": nothing here trains. ppo_tester only runs frozen
# policies deterministically - it has no optimiser, no gradient step, and the
# models are loaded read-only. No training job is running concurrently either;
# the last chain (p75p75-d3) is queued behind this and in any case writes to
# its own directory.
#
# TIME: measured 60-73 s per episode-worker across three prior evals. 6,000
# episodes x ~66 s / 64 workers ~= 1.7 h on an owl0xx node, ~2.4 h on a slow
# hm node. 6 h wall gives 2.5x margin over the worst case.
MODELS=(final_evaluation/models/arm1_palisades_only.zip
        final_evaluation/models/arm2_pyrenees_only.zip
        final_evaluation/models/arm3_even_split_75_75.zip
        final_evaluation/models/arm4_pal25k_pyr125k.zip
        final_evaluation/models/arm5_concurrent.zip
        final_evaluation/models/arm6_palisades_multiweight.zip)

OUTDIR=final_evaluation
mkdir -p "$OUTDIR"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  echo "########## $SC ##########"
  python -u -m examples.wildfire.ppo_tester \
    --model-paths "${MODELS[@]}" \
    --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
    --include-random-tactics \
    --scenarios "$SC" \
    --runs-per-scenario 250 \
    --seed 1 --random-seeds \
    --device cpu --num-workers 64 \
    --state-space directional-2 --state-fire-fronts 3 \
    --water-set 2 --switch-ignition 4 \
    --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
    --output "$OUTDIR/results_${TAG}.csv" \
    --summary-output "$OUTDIR/summary_${TAG}.csv" \
    --plots-dir "$OUTDIR/plots_${TAG}"
done

# Merge and VERIFY the pairing rather than assume it.
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

# Statistics: Wilson CIs, exact McNemar, paired t + Wilcoxon, Holm-corrected.
{
  echo "FINAL EVALUATION - 8 policies x 3 maps x 250 paired ignitions"
  echo "generated $(date -Is)"; echo
  python -u scripts/eval_stats.py "$OUTDIR/results.csv"
  echo; echo "### every policy against the Pyrenees-only arm ###"
  python -u scripts/eval_stats.py "$OUTDIR/results.csv" --reference arm2_pyrenees_only
} | tee "$OUTDIR/statistical_comparison.txt"
echo "=== final_evaluation populated ==="
ls -l "$OUTDIR"
