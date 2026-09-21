#!/usr/bin/env bash
#SBATCH -J eval-th-pyr-1k
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 04:00:00
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

# PRE-SPECIFIED CONFIRMATORY EVALUATION, Pyrenees sibling of 894456.
# Committed before data collection. Fresh --seed 2 draw (independent of
# every seed-1 evaluation; note the Palisades 894456 fires are a different
# map, so no overlap in fire instances either).
#
# HYPOTHESES, fixed in advance (matching what all prior seed-1 draws say
# about Pyrenees - we pre-specify what we EXPECT, not what we wish):
#   H1 (directional): arm5_concurrent > arm1_palisades_only on mean MoE.
#       Prior: +6.1 sigma at n=250. One-sided alpha 0.05.
#   E1 (equivalence bound, NOT a superiority test): arm5_concurrent vs
#       arm2_pyrenees_only - two-sided test plus 95% CI on the paired MoE
#       difference. Prior: +0.5 sigma, below the 0.7-sigma noise floor;
#       expectation is a tie, and the CI documents how tight.
# Secondary endpoints: containment (exact McNemar), same structure.
MODELS=(final_evaluation/models/arm5_concurrent.zip
        final_evaluation/models/arm2_pyrenees_only.zip
        final_evaluation/models/arm1_palisades_only.zip)

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_theater_pyrenees_1000sims_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# 3 policies x 1000 episodes; Pyrenees episodes end early (escapes), so
# runtime is at or below the ~52 min the Palisades sibling took.
python -u -m examples.wildfire.ppo_tester \
  --model-paths "${MODELS[@]}" \
  --scenarios Pyrenees5sp5ev.json \
  --runs-per-scenario 1000 \
  --seed 2 --random-seeds \
  --device cpu --num-workers 64 \
  --state-space directional-2 --state-fire-fronts 3 \
  --water-set 2 --switch-ignition 4 \
  --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"

python - "$OUTDIR" <<'PYEOF' | tee "$OUTDIR/prespecified_analysis.txt"
import sys, numpy as np, pandas as pd
from scipy import stats
out = sys.argv[1]
df = pd.read_csv(f"{out}/results.csv")
df["moe"] = pd.to_numeric(df["moe_cumulative_reward"])
df["cont"] = pd.to_numeric(df["propagation_factor"]) != 1.0
pm = df.pivot_table(index="seed", columns="policy_name", values="moe")
pc = df.pivot_table(index="seed", columns="policy_name", values="cont").astype(bool)
assert pm.notna().all().all(), "unpaired rows"
T = "arm5_concurrent"
print(f"PRE-SPECIFIED ANALYSIS - Pyrenees, n={len(pm)} paired ignitions, seed base 2")
print(f"{'policy':28} contained   mean MoE")
for p in pm.columns:
    print(f"{p:28} {pc[p].mean()*100:7.1f}%   {pm[p].mean():+.4f}")
# H1: directional superiority over the off-map specialist
o = "arm1_palisades_only"
d = pm[T] - pm[o]
p1 = stats.ttest_rel(pm[T], pm[o], alternative="greater").pvalue
n10 = int((pc[T] & ~pc[o]).sum()); n01 = int((~pc[T] & pc[o]).sum())
pmc = stats.binomtest(n10, n10 + n01, alternative="greater").pvalue if n10 + n01 else 1.0
sg = d.mean() / d.sem()
verdict = "SUPPORTED" if p1 < 0.05 and abs(sg) >= 0.7 else "not supported"
print(f"H1 theater > {o}: dMoE {d.mean():+.4f}  sigma {sg:+.2f}  "
      f"1-sided p {p1:.4g}  McNemar {n10}:{n01} p {pmc:.4g}  -> {verdict}")
# E1: equivalence bound vs the native specialist
o = "arm2_pyrenees_only"
d = pm[T] - pm[o]
p2 = stats.ttest_rel(pm[T], pm[o]).pvalue
ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.sem())
n10 = int((pc[T] & ~pc[o]).sum()); n01 = int((~pc[T] & pc[o]).sum())
pmc = stats.binomtest(min(n10, n01), n10 + n01).pvalue if n10 + n01 else 1.0
sg = d.mean() / d.sem()
tie = "TIE (below 0.7-sigma noise floor)" if abs(sg) < 0.7 else (
    "difference detected" if p2 < 0.05 else "inconclusive")
print(f"E1 theater vs {o}: dMoE {d.mean():+.4f}  95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]  "
      f"sigma {sg:+.2f}  2-sided p {p2:.4g}  McNemar {n10}:{n01} p {pmc:.4g}  -> {tie}")
PYEOF

python -u scripts/plot_eval2_horizontal.py "$OUTDIR/results.csv" _1000sims
echo "=== done: $OUTDIR ==="
