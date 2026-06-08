#!/usr/bin/env bash
#SBATCH -J pyr-wcmp
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=300G
#SBATCH --qos=owl_normal_short

# Pyrenees: REGULAR water (--water-set off, 140 bodies) vs SMALLEST water
# (--water-set 1, 19 bodies). One run each, back-to-back on the same node so
# timings are directly comparable. Identical settings except --water-set.

set -uo pipefail   # NOT -e: first run failing must not skip the second

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# === Thread pinning (40 envs on 80 cores; avoid BLAS oversubscription) ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Short, identical budget so the two runs finish and are comparable.
# 28800 / 40 envs / 144 n_steps = 5 rollout iterations (past JIT warmup).
TIMESTEPS=28800

run_one () {
  local ws="$1"           # "off" or "1"
  local tag="pyr_wcmp_${ws}"
  local outdir="examples/wildfire/data/scenarios/outputs/${tag}_${SLURM_JOB_ID}"
  local runlog="logs/${tag}_${SLURM_JOB_ID}.log"
  echo ""
  echo "########## Pyrenees  water-set=${ws} ##########"
  SECONDS=0
  /usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
    --scenario "Pyrenees.json" \
    --num-envs 40 \
    --timesteps "$TIMESTEPS" \
    --device cpu \
    --output-dir "$outdir" \
    --progress-file-tag "$tag" \
    --fire-detection-delay-minutes 30 \
    --vec-start-method auto \
    --state-fire-fronts 5 \
    --gc-collect-on-reset \
    --n-epochs 7 \
    --learning-rate 0.0004 \
    --gamma 0.99 \
    --state-space mixed \
    --tactic-distribution group \
    --aircraft-group-size 4 \
    --controlled-agent-count 12 \
    --water-set "$ws" \
    --switch-ignition-2 \
    > "$runlog" 2>&1
  echo "RESULT water-set=${ws} exit=$? wall=${SECONDS}s  log=${runlog}"
}

echo "=========== PYRENEES WATER COMPARISON (timesteps=$TIMESTEPS) ==========="
run_one off    # regular: 140 water bodies
run_one 1      # smallest: 19 water bodies

echo ""
echo "=================== TIMING SUMMARY ==================="
for ws in off 1; do
  rl="logs/pyr_wcmp_${ws}_${SLURM_JOB_ID}.log"
  fps=$(grep -oE 'fps *\| *[0-9]+' "$rl" | tail -1 | grep -oE '[0-9]+$')
  te=$(grep -oE 'time_elapsed *\| *[0-9]+' "$rl" | tail -1 | grep -oE '[0-9]+$')
  ts=$(grep -oE 'total_timesteps *\| *[0-9]+' "$rl" | tail -1 | grep -oE '[0-9]+$')
  printf "  water-set=%-3s : fps=%-4s time_elapsed=%-6s total_timesteps=%-6s\n" \
    "$ws" "${fps:-?}" "${te:-?}" "${ts:-?}"
done
echo "====================================================="
