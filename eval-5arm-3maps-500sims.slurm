#!/usr/bin/env bash
#SBATCH -J eval-5arm-3maps
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

OUTDIR=examples/wildfire/data/scenarios/outputs/eval_5arm_3maps_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

# The five ~100k curriculum arms on THREE maps, 500 held-out ignitions each:
# 5 policies x 3 maps x 500 = 7,500 episodes, one --seed 1 --random-seeds draw
# so every policy sees the same 500 ignitions per map (fully paired).
#
# SALAMIS IS NEW HERE. No arm has ever trained on it - the training set is
# Palisades and Pyrenees only - so for every one of the five policies Salamis is
# a ZERO-SHOT third map. That makes it the cleanest generalisation test in the
# study: Palisades and Pyrenees results are partly in-distribution, Salamis is
# not, for anyone.
#
# Salamis5sp5ev.json was created for this run from Salamis6sp6ev.json, keeping
# the map's one-base-per-type convention (seaplanes [5,0], eVTOLs [0,5]) and
# taking the suppression_tactic block from Pyrenees5sp5ev.json so it matches the
# other 5sp5ev scenarios. Verified it builds and that _resolve_scenario_one_hot
# returns (0,0,1) for it, so the three multi-mission arms that carry the
# scenario one-hot get a valid Salamis encoding rather than the all-zeros
# fallback an unrecognised name would produce.
python -u -m examples.wildfire.ppo_tester \
  --model-paths eval_models_100k/arm1_palisades_only_93k8.zip \
                eval_models_100k/arm2_pyrenees_only_100k.zip \
                eval_models_100k/arm3_even_split_seq_100k.zip \
                eval_models_100k/arm4_pal25k_pyr125k_97k.zip \
                eval_models_100k/arm5_concurrent_100k.zip \
  --scenarios Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json \
  --runs-per-scenario 500 \
  --seed 1 --random-seeds \
  --device cpu --num-workers 96 \
  --state-space directional-2 --state-fire-fronts 3 \
  --water-set 2 --switch-ignition 4 \
  --group-sizes 5 5 --controlled-agent-count 10 --cell-size code \
  --output "$OUTDIR/results.csv" \
  --summary-output "$OUTDIR/summary.csv" \
  --plots-dir "$OUTDIR/plots"

for SC in Palisades5sp5ev.json Pyrenees5sp5ev.json Salamis5sp5ev.json; do
  TAG=$(echo "${SC%%5sp5ev.json}" | tr 'A-Z' 'a-z')
  python -u scripts/plot_eval_boxplots.py "$OUTDIR/results.csv" \
    --scenario "$SC" --color-by-type \
    --order arm1_palisades_only_93k8 arm2_pyrenees_only_100k \
            arm3_even_split_seq_100k arm4_pal25k_pyr125k_97k arm5_concurrent_100k \
    --labels "1 Palisades only" "2 Pyrenees only" "3 even split 50/50" \
             "4 Pal 25k→Pyr" "5 concurrent" \
    --note "Five curriculum arms at ~100k · 500 paired held-out ignitions · Salamis is zero-shot for every arm" \
    --title "${SC%%5sp5ev.json} · five curricula · 500 paired simulations" \
    --out "verification_plots/eval_3map_${TAG}.png"
done

REPORT="$OUTDIR/statistical_comparison.txt"
{
  echo "Five curriculum arms, three maps, 500 paired ignitions each"
  echo "generated $(date -Is)"
  echo
  python -u scripts/eval_stats.py "$OUTDIR/results.csv"
  echo
  echo "### every arm against the Pyrenees-only control ###"
  python -u scripts/eval_stats.py "$OUTDIR/results.csv" --reference arm2_pyrenees_only_100k
} | tee "$REPORT"
echo "=== results: $OUTDIR ==="
