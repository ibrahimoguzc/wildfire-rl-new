#!/usr/bin/env bash
#SBATCH -J eval-5arm-500
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

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_5arm_100k_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# The five curriculum arms, all at ~100k simulations, on 500 held-out ignitions
# per map. Arms 1 and 3 have no 150k model yet (still training), so the whole
# set is taken at the ~100k mark to keep the budgets comparable:
#   arm1 Palisades only        93,800 sims  (chain stopped early on MAX_CHUNKS)
#   arm2 Pyrenees only        100,005 sims  (chunk15 of the original chain)
#   arm3 even split 50/50     100,000 sims  (job 679425 - an exact 50k/50k split)
#   arm4 Pal 25k -> Pyr       ~97,160 sims  (717320 checkpoint @1,763,328 steps)
#   arm5 concurrent 64/64    ~100,345 sims  (chunk16 of the concurrent chain)
#
# Budget spread is 93,800-100,345, about 7%. Arm 1 is the outlier and its curve
# was flat from ~60,000, so the gap should not matter - but it is the one number
# to caveat if arm 1 looks weak.
#
# NOT all on the same LR schedule. Arms 1, 2 and 3 ran 100k-budget schedules and
# are fully annealed at this point; arms 4 and 5 are 150k-budget runs caught
# two thirds of the way through, so their LR was still ~1.8e-04 when the
# checkpoint was taken. An annealed policy is usually a little sharper, so arms
# 4 and 5 are mildly disadvantaged here. This resolves itself once their 150k
# finals are used.
#
# Observation widths differ (66 for the single-map arms, 69 for the three
# multi-mission ones); ppo_tester detects env_obs + len(SCENARIO_FEATURES) ==
# model_obs and rebuilds the env with include_scenario_features for those.
#
# 500 runs x 2 maps x 5 policies = 5,000 episodes. One invocation, one
# --seed 1 --random-seeds draw, so every policy sees the same 500 ignitions per
# map and all comparisons are paired.
python -u -m examples.wildfire.ppo_tester \
  --model-paths eval_models_100k/arm1_palisades_only_93k8.zip \
                eval_models_100k/arm2_pyrenees_only_100k.zip \
                eval_models_100k/arm3_even_split_seq_100k.zip \
                eval_models_100k/arm4_pal25k_pyr125k_97k.zip \
                eval_models_100k/arm5_concurrent_100k.zip \
  --scenarios Pyrenees5sp5ev.json Palisades5sp5ev.json \
  --runs-per-scenario 500 \
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
    --order arm1_palisades_only_93k8 arm2_pyrenees_only_100k \
            arm3_even_split_seq_100k arm4_pal25k_pyr125k_97k arm5_concurrent_100k \
    --labels "1 Palisades only" "2 Pyrenees only" "3 even split 50/50" \
             "4 Pal 25k→Pyr" "5 concurrent" \
    --note "All five curriculum arms at ~100k simulations · 500 paired held-out ignitions" \
    --title "${SC%%5sp5ev.json} · five curricula at ~100k · 500 paired simulations" \
    --out "verification_plots/eval_5arm_100k_${TAG}.png"
done
echo "=== results: $OUTDIR ==="
