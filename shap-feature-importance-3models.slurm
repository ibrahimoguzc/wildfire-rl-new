#!/usr/bin/env bash
#SBATCH -J shap-3models
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
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

OUT=examples/wildfire/data/scenarios/outputs/shap_${SLURM_JOB_ID}
mkdir -p "$OUT"

# SHAP feature importance for three trained policies, all evaluated on Pyrenees
# under the ignition distribution they trained on (switch-ignition-4).
#
# Only 16 CPUs: the script's rollout loop is single-threaded (one env, stepped
# serially), so a whole node would sit idle. The three models are launched
# CONCURRENTLY instead - that is the only parallelism available here, and it
# turns ~3x50 min of serial work into one ~60-90 min wall-clock block.
#
# Why three models rather than one:
#   717319  Pyrenees-only 150k          66 features, NO scenario one-hot
#   717320  Pal 25k -> Pyr 125k 150k    69 features, HAS the one-hot
#   717323  concurrent 64/64 150k       69 features, HAS the one-hot
# The pair of 69-feature models is what answers "does the policy actually use
# the scenario flags, or are they dead weight?" - a question the single-map
# model cannot answer because it does not have them. Comparing 717319 against
# the other two also shows whether importance rankings are a property of the
# task or of the curriculum.
#
# --episodes 60 / --explain-size 1000 / --background-size 100 is roughly double
# the interactive attempt. Rankings from a few hundred on-policy states are
# noisy at the tail; the head of the ranking stabilises much faster than the
# tail, and it is the tail ("what can I delete?") that this analysis is for.

run_one () {
  local tag=$1 model=$2 agents=$3
  ~/.conda/envs/rl-env/bin/python -u scripts/shap_feature_importance.py \
    --model-path "$model" \
    --scenario "Pyrenees5sp5ev.json" \
    --switch-ignition-mode 4 \
    --state-space directional-2 \
    --state-fire-fronts 3 \
    --controlled-agent-count "$agents" \
    --episodes 60 \
    --explain-size 1000 \
    --background-size 100 \
    --device cpu \
    --output-dir "$OUT/$tag" \
    > "$OUT/${tag}.log" 2>&1 &
}

O=examples/wildfire/data/scenarios/outputs
run_one pyr_only_717319 \
  "$O/pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_150k_717319/pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip" 10
run_one pal25_pyr125_717320 \
  "$O/missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_717320/missions_pal25k_pyr125k_5sp5ev_grpbytype_sw4_directional2_k3_sims150000.zip" 10
run_one concurrent_717323 \
  "$O/concurrent_palisades_pyrenees_5sp5ev_grpbytype_sw4_directional2_k3_chain/latest.zip" 10

wait
echo "=== all three finished; results under $OUT ==="
for d in "$OUT"/*/; do echo "--- $d"; ls "$d" 2>/dev/null | head; done
