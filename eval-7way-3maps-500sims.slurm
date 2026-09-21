#!/usr/bin/env bash
#SBATCH -J eval-7way-3maps
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH -t 08:00:00
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

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_7way_3maps_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# 7 policies x 3 maps x 500 held-out ignitions = 10,500 episodes.
#
# REWRITTEN AFTER AN OOM. Job 753124 ran all three maps in ONE ppo_tester
# invocation with 96 workers and was OOM-killed at 8,228/10,500: MaxRSS
# 821 GB against the node's 761 GB, ~8.6 GB per worker. A single env measures
# 1.5-2 GB, so the excess is growth during long episodes - fire grids expand as
# a fire spreads, and the arms that contain least (random tactics) burn the
# most. It died 78% in, which fits memory creeping up rather than a fixed
# overflow. ppo_tester writes its CSV only at the end, so all 8,228 episodes
# were lost.
#
# Two changes:
#   1. ONE INVOCATION PER MAP. Each is a fresh process, so memory is released
#      between maps instead of accumulating across all three. It also caps the
#      blast radius: a failure now costs one map, not the whole run.
#   2. 64 workers instead of 96 - two thirds the peak, ~547 GB by the observed
#      per-worker figure, a real margin under 761 GB.
#
# Cost: ~55 min per map instead of ~45, so ~3 h total rather than ~2.3 h. The
# 8 h wall covers a slow node (753124 landed on owl-hm001, ~30% slower than the
# owl0xx machines).
#
# Each invocation uses the same --seed 1 --random-seeds, so every policy sees
# the identical 500 ignitions within a map and all within-map comparisons stay
# paired. The three CSVs are concatenated afterwards for the pooled figures.
MODELS=(eval_models_100k/arm1_palisades_only_93k8.zip
        eval_models_100k/arm2_pyrenees_only_100k.zip
        eval_models_100k/arm3_even_split_seq_100k.zip
        eval_models_100k/arm4_pal25k_pyr125k_97k.zip
        eval_models_100k/arm5_concurrent_100k.zip)

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  echo "########## $SC ##########"
  python -u -m examples.wildfire.ppo_tester \
    --model-paths "${MODELS[@]}" \
    --fixed-tactic-scenarios Pyrenees5sp5ev_heuristic.json \
    --include-random-tactics \
    --scenarios "$SC" \
    --runs-per-scenario 500 \
    --seed 1 --random-seeds \
    --device cpu --num-workers 64 \
    --state-space directional-2 --state-fire-fronts 3 \
    --water-set 2 --switch-ignition 4 \
    --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
    --output "$OUTDIR/results_${TAG}.csv" \
    --summary-output "$OUTDIR/summary_${TAG}.csv" \
    --plots-dir "$OUTDIR/plots_${TAG}"
done

# Merge the three per-map CSVs into one for the pooled figures and the stats.
python - "$OUTDIR" <<'PYEOF'
import sys, glob, pandas as pd
out = sys.argv[1]
parts = sorted(glob.glob(f"{out}/results_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv(f"{out}/results.csv", index=False)
print(f"merged {len(parts)} maps -> {len(df):,} rows")
print(df.groupby("scenario_name")["policy_name"].nunique().to_string())
PYEOF

ORDER=(random_tactics fixed_Pyrenees5sp5ev_heuristic
       arm1_palisades_only_93k8 arm2_pyrenees_only_100k
       arm3_even_split_seq_100k arm4_pal25k_pyr125k_97k arm5_concurrent_100k)
LABELS=("Random" "Fixed tactics" "1 Palisades only" "2 Pyrenees only"
        "3 even split 50/50" "4 Pal 25k→Pyr" "5 concurrent")
NOTE="Grey = non-learned baseline · 500 paired held-out ignitions per map · Salamis is zero-shot for every trained arm"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
    --scenario "$SC" --color-by-type --order "${ORDER[@]}" --labels "${LABELS[@]}" \
    --note "$NOTE" --title "${SC%%5sp5ev.json} · 7 policies · 500 paired simulations" \
    --out "verification_plots/eval7_${TAG}.png"
done

python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" --color-by-type \
  --order "${ORDER[@]}" --labels "${LABELS[@]}" \
  --note "Pooled over Palisades + Pyrenees + Salamis, 500 ignitions each (equal weight)" \
  --title "Pooled across all three maps · 1500 paired simulations per policy" \
  --out verification_plots/eval7_pooled_3maps.png

python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" --color-by-type \
  --scenarios Palisades5sp5ev.json Pyrenees5sp5ev.json \
  --order "${ORDER[@]}" --labels "${LABELS[@]}" \
  --note "Pooled over the two TRAINED maps only, 500 ignitions each (equal weight)" \
  --title "Pooled across Palisades + Pyrenees · 1000 paired simulations per policy" \
  --out verification_plots/eval7_pooled_2maps.png

REPORT="$OUTDIR/statistical_comparison.txt"
{
  echo "7 policies x 3 maps x 500 paired ignitions"; echo "generated $(date -Is)"; echo
  python -u scripts/eval_stats.py "$OUTDIR/results.csv"
  echo; echo "### every policy against the Pyrenees-only arm ###"
  python -u scripts/eval_stats.py "$OUTDIR/results.csv" --reference arm2_pyrenees_only_100k
} | tee "$REPORT"
echo "=== results: $OUTDIR ==="
