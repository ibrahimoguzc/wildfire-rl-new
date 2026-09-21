#!/usr/bin/env bash
#SBATCH -J eval-th-pal-1k
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

# PRE-SPECIFIED CONFIRMATORY EVALUATION - committed before data collection.
#
# Motivation: at n=250 (final evaluation, seed 1) the theater-level agent
# led both specialists on Palisades by ~2 sigma - above the 0.7-sigma
# replicate noise floor but only marginal after correction (one-sided
# Holm p = 0.044 / 0.046). This job is the powered, independent test.
#
# HYPOTHESES (directional, fixed in advance, Holm across the two):
#   H1: arm5_concurrent > arm2_pyrenees_only   on Palisades (mean MoE)
#   H2: arm5_concurrent > arm1_palisades_only  on Palisades (mean MoE)
# Secondary endpoints, same directions: containment (exact McNemar).
#
# POWER: observed paired d = 0.128 (H1) / 0.107 (H2). At n=1000, one-sided
# alpha 0.05/2, power ~ 0.98 (H1) / 0.93 (H2). If an effect this size is
# real, this run finds it; if the run comes back null, the Palisades edge
# joins the noise-floor bucket and the paper claims the tie.
#
# INDEPENDENCE: --seed 2. Every prior evaluation drew from the seed-1
# stream (the 250/100/50 sets are nested prefixes of it); seed 2 draws a
# fresh, non-overlapping ignition set, so this is a true replication.
MODELS=(final_evaluation/models/arm5_concurrent.zip
        final_evaluation/models/arm2_pyrenees_only.zip
        final_evaluation/models/arm1_palisades_only.zip)

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_theater_palisades_1000sims_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# 3 policies x 1000 episodes ~= 3,000 x 66 s / 64 workers ~= 52 min.
python -u -m examples.wildfire.ppo_tester \
  --model-paths "${MODELS[@]}" \
  --scenarios Palisades5sp5ev.json \
  --runs-per-scenario 1000 \
  --seed 2 --random-seeds \
  --device cpu --num-workers 64 \
  --state-space directional-2 --state-fire-fronts 3 \
  --water-set 2 --switch-ignition 4 \
  --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"

# The pre-specified analysis, exactly as stated above.
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
print(f"PRE-SPECIFIED ANALYSIS - Palisades, n={len(pm)} paired ignitions, seed base 2")
print(f"{'policy':28} contained   mean MoE")
for p in pm.columns:
    print(f"{p:28} {pc[p].mean()*100:7.1f}%   {pm[p].mean():+.4f}")
res = []
for o in ("arm2_pyrenees_only", "arm1_palisades_only"):
    d = pm[T] - pm[o]
    p1 = stats.ttest_rel(pm[T], pm[o], alternative="greater").pvalue
    n10 = int((pc[T] & ~pc[o]).sum()); n01 = int((~pc[T] & pc[o]).sum())
    pmc = stats.binomtest(n10, n10 + n01, alternative="greater").pvalue if n10 + n01 else 1.0
    res.append((o, d.mean(), d.mean() / d.sem(), p1, n10, n01, pmc))
pv = sorted(range(2), key=lambda i: res[i][3])
holm = {}
prev = 0.0
for rank, i in enumerate(pv):
    holm[i] = min(1.0, max(prev, (2 - rank) * res[i][3])); prev = holm[i]
for i, (o, dm, sg, p1, n10, n01, pmc) in enumerate(res):
    verdict = "SUPPORTED" if holm[i] < 0.05 and abs(sg) >= 0.7 else "not supported"
    print(f"H: theater > {o:22} dMoE {dm:+.4f}  sigma {sg:+.2f}  "
          f"1-sided p {p1:.4g}  Holm(2) {holm[i]:.4g}  "
          f"McNemar {n10}:{n01} p {pmc:.4g}  -> {verdict}")
PYEOF
echo "=== done: $OUTDIR ==="
