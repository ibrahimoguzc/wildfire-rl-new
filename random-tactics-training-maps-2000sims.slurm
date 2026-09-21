#!/usr/bin/env bash
#SBATCH -J rand-train-maps
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
#SBATCH -x owl-hm[001-003]

set -euo pipefail
mkdir -p logs
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Random-tactics baseline for the TRAINING curves (final_plots/training):
# 2,000 simulations on each training map - Palisades and Pyrenees - under the
# exact regime the three training instances saw (switch-ignition-4, water set
# 2, directional-2 K=3, 5sp+5ev grouped by type, code-derived cell size).
#
# 2,000 per map matches the 2,000-episode moving-average window of the
# training figures, so the baseline mean rests on the same sample size as any
# single point of those curves. The per-episode ignitions of the actual
# training runs are not replayable through ppo_tester (it draws its own seed
# list), and a matched-regime sample estimates the same quantity: random
# tactics do not learn, so their curve is flat at this mean.
#
# No model is loaded and nothing trains: --include-random-tactics alone is
# the whole policy list, and the tester samples a fresh tactic tuple per
# group from the env's own action space at every decision step.
#
# One invocation per map (the post-OOM convention from the 8-way final eval);
# 64 workers. TIME: 4,000 episodes x ~66 s / 64 workers ~= 1.2 h measured on
# prior evals; random episodes end early (escapes), so 4 h wall is ample.
OUTDIR=examples/wildfire/data/scenarios/outputs/random_tactics_training_maps_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  echo "########## $SC ##########"
  python -u -m examples.wildfire.ppo_tester \
    --include-random-tactics \
    --scenarios "$SC" \
    --runs-per-scenario 2000 \
    --seed 1 --random-seeds \
    --device cpu --num-workers 64 \
    --state-space directional-2 --state-fire-fronts 3 \
    --water-set 2 --switch-ignition 4 \
    --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
    --output "$OUTDIR/results_${TAG}.csv" \
    --summary-output "$OUTDIR/summary_${TAG}.csv" \
    --plots-dir "$OUTDIR/plots_${TAG}"
done

# Headline numbers into the log: mean MoE and escape rate per map.
python - "$OUTDIR" <<'PYEOF'
import sys, glob, pandas as pd
out = sys.argv[1]
for p in sorted(glob.glob(f"{out}/results_*.csv")):
    df = pd.read_csv(p)
    esc = (pd.to_numeric(df["propagation_factor"]) == 1.0).mean() * 100.0
    moe = pd.to_numeric(df["moe_cumulative_reward"]).mean()
    print(f"{p}: n={len(df)}  mean MoE {moe:+.4f}  escape {esc:.1f}%")
PYEOF
echo "=== done: $OUTDIR ==="
