#!/usr/bin/env bash
#SBATCH -J water-cmp
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --qos=owl_normal_short

# Compare LOW water resources (--water-set 1) vs REGULAR (--water-set off)
# across all three maps. Each (map x water-set) is a short, identical training
# run, so wall-time / fps are directly comparable and the summary CSVs capture
# the outcome effect. 6 runs total, sequential.

set -uo pipefail   # NOT -e: one run failing must not kill the rest

mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# === Thread pinning (36 envs on 72 cores; avoid BLAS oversubscription) ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Short, identical budget for every run so timings are comparable.
# 25920 / 36 envs / 144 n_steps = 5 rollout iterations (enough for fps to settle
# past JIT warmup). Bump TIMESTEPS for a longer learning comparison.
TIMESTEPS=25920
NUM_ENVS=36
DELAY=30          # fire-detection delay (min); same for all runs
SEED=0

# Per-map fleet config (controlled aircraft / group size). Must satisfy
# controlled-agent-count <= scenario fleet (Pyrenees 14, Palisades 6, Salamis 3).
MAPS=(pyrenees palisades salamis)
declare -A SCEN=( [pyrenees]="Pyrenees.json" [palisades]="Palisades copy.json" [salamis]="Salamis.json" )
declare -A CTRL=( [pyrenees]=12 [palisades]=6 [salamis]=3 )
declare -A GRP=(  [pyrenees]=4  [palisades]=2 [salamis]=1 )

echo "================ WATER COMPARISON ================"
echo "timesteps=$TIMESTEPS num_envs=$NUM_ENVS delay=${DELAY}min seed=$SEED"
echo "================================================="

for map in "${MAPS[@]}"; do
  for ws in off 1; do
    TAG="watercmp_${map}_${ws}"
    OUTDIR="examples/wildfire/data/scenarios/outputs/${TAG}_${SLURM_JOB_ID}"
    RUNLOG="logs/${TAG}_${SLURM_JOB_ID}.log"
    echo ""
    echo "########## RUN: map=${map} water-set=${ws} ##########"
    SECONDS=0
    /usr/bin/time -v python -u -m examples.wildfire.ppo_runnerv2 \
      --scenario "${SCEN[$map]}" \
      --num-envs "$NUM_ENVS" \
      --timesteps "$TIMESTEPS" \
      --device cpu \
      --output-dir "$OUTDIR" \
      --progress-file-tag "$TAG" \
      --fire-detection-delay-minutes "$DELAY" \
      --vec-start-method auto \
      --state-fire-fronts 5 \
      --gc-collect-on-reset \
      --n-epochs 7 \
      --learning-rate 0.0004 \
      --gamma 0.99 \
      --state-space mixed \
      --tactic-distribution group \
      --aircraft-group-size "${GRP[$map]}" \
      --controlled-agent-count "${CTRL[$map]}" \
      --water-set "$ws" \
      --switch-ignition-2 \
      > "$RUNLOG" 2>&1
    rc=$?
    echo "RESULT map=${map} water-set=${ws} exit=${rc} wall=${SECONDS}s  log=${RUNLOG}"
  done
done

echo ""
echo "================ TIMING SUMMARY ================"
# Pull the last reported fps / time_elapsed / total_timesteps from each run log.
for map in "${MAPS[@]}"; do
  for ws in off 1; do
    RUNLOG="logs/watercmp_${map}_${ws}_${SLURM_JOB_ID}.log"
    fps=$(grep -oE 'fps *\| *[0-9]+' "$RUNLOG" | tail -1 | grep -oE '[0-9]+$')
    te=$(grep -oE 'time_elapsed *\| *[0-9]+' "$RUNLOG" | tail -1 | grep -oE '[0-9]+$')
    ts=$(grep -oE 'total_timesteps *\| *[0-9]+' "$RUNLOG" | tail -1 | grep -oE '[0-9]+$')
    printf "  %-10s water-set=%-3s : fps=%-5s time_elapsed=%-7s total_timesteps=%-7s\n" \
      "$map" "$ws" "${fps:-?}" "${te:-?}" "${ts:-?}"
  done
done
echo "==============================================="
