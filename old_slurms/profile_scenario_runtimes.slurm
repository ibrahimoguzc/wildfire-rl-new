#!/usr/bin/env bash
#
# Profile per-scenario PPO training runtimes to diagnose why
# Palisades trains much faster than Pyrenees and Salamis.
#
# Phases:
#   1) One short PPO run per scenario in solo mode with --profile-timings.
#      Each run prints a per-rollout table with step_mean / step_p95 /
#      reset_mean / free_burn_mean / RSS, broken down by scenario_name.
#   2) A combined switch-scenario run with the same flags, so step-time
#      ratios are observed under the same OS/cache conditions in one process.
#   3) A pyinstrument trace on a single headless episode per scenario
#      (examples.wildfire.main) for function-level hot spots.
#
# Look for these in logs/slurm-<jobid>.out:
#   * "[profile-timings] rollout N summary:" tables
#   * "Decision start delay:" lines per scenario
#   * pyinstrument top frames (e.g. fire_model.cpu.step,
#     FirefighterModel.step, cKDTree.query, distance_transform_edt)
#
#SBATCH -J profile-rl
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm-%j.out
#SBATCH -e logs/slurm-%j.err
#SBATCH --mem=120G
#SBATCH --qos=owl_normal_short
mkdir -p logs

# === Environment Setup ===
module reset
module load Miniforge3
eval "$(conda shell.bash hook)"
source activate rl-env
which python

cd /home/ibrahimoguz/FirefightingSoS/wildfire-rl-new

# Keep workers identical across phases so step-time deltas reflect the
# scenario, not vec-env contention. NUM_ENVS * STEPS_PER_ENV must give us
# at least 2-3 PPO rollouts per scenario (rollout = 144 steps/env).
NUM_ENVS=12
TIMESTEPS=10368   # 12 envs * 144 n_steps * 6 rollouts

OUTDIR_BASE=examples/wildfire/data/scenarios/outputs/profile_$SLURM_JOB_ID
mkdir -p "$OUTDIR_BASE"

run_solo () {
  local scenario_json="$1"
  local tag="$2"
  echo "============================================================"
  echo ">>> SOLO PROFILE: $scenario_json"
  echo "============================================================"
  /usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
    --scenario "$scenario_json" \
    --switch-ignition-1 \
    --num-envs "$NUM_ENVS" \
    --timesteps "$TIMESTEPS" \
    --device cpu \
    --profile-timings \
    --gc-collect-on-reset \
    --output-dir "$OUTDIR_BASE/solo_${tag}" \
    --progress-file-tag "profile_${tag}" \
    --vec-start-method auto
}

# Phase 1: solo runs (one scenario per process). This isolates each
# scenario's per-step / per-reset cost from any switch-scheduling effects.
# Note: use "Palisades copy.json" (the in-bounds, codebase-default file).
# Bare "Palisades.json" has urban_locations that sit on the operational
# bbox corners and fail strict pydantic validation by ~4e-9 degrees.
run_solo "Palisades copy.json" "palisades"
run_solo "Pyrenees.json"        "pyrenees"
run_solo "Salamis.json"         "salamis"

# Phase 2: switch-scenario run, all three round-robin assigned to workers.
# Steps from each scenario interleave in the same Python process, so the
# per-rollout summary table is a direct apples-to-apples comparison.
echo "============================================================"
echo ">>> SWITCH-SCENARIO PROFILE (round-robin Palisades/Pyrenees/Salamis)"
echo "============================================================"
/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --switch-scenario \
  --switch-scenarios "Palisades copy.json" "Pyrenees.json" "Salamis.json" \
  --switch-ignition-1 \
  --num-envs "$NUM_ENVS" \
  --timesteps "$TIMESTEPS" \
  --device cpu \
  --profile-timings \
  --gc-collect-on-reset \
  --output-dir "$OUTDIR_BASE/switch_all" \
  --progress-file-tag "profile_switch" \
  --vec-start-method auto

# Phase 3: function-level profile of a single headless episode per scenario
# using pyinstrument (already used by examples/wildfire/main.py when
# RUN_PROFILER=True). We invoke a small inline driver so we don't have to
# edit main.py, and dump the HTML/text profile under $OUTDIR_BASE.
echo "============================================================"
echo ">>> PYINSTRUMENT PER-SCENARIO HEADLESS RUN"
echo "============================================================"

PYPROFILE_DIR="$OUTDIR_BASE/pyinstrument"
mkdir -p "$PYPROFILE_DIR"
export PYPROFILE_DIR

for SCENARIO in "Palisades copy.json" "Pyrenees.json" "Salamis.json"; do
  TAG=$(echo "$SCENARIO" | sed 's/\.json$//' | tr '[:upper:] ' '[:lower:]_')
  echo "--- pyinstrument: $SCENARIO ---"
  SCEN="$SCENARIO" TAG="$TAG" python -u - <<'PYEOF'
import os, time
from pathlib import Path
from pyinstrument import Profiler

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation

scen = os.environ["SCEN"]
tag  = os.environ["TAG"]
outdir = Path(os.environ["PYPROFILE_DIR"])
outdir.mkdir(parents=True, exist_ok=True)

scenario_path = SCENARIOS_DIR / "inputs" / scen
print(f"Profiling headless episode for {scen}")
with scenario_path.open() as f:
    params = WildfireParameters.model_validate_json(f.read())

# Cap wall-clock budget so a Pyrenees mega-fire doesn't hog the node;
# fire model still reports steps until extinguished or this fires.
WALL_BUDGET_S = 600

prof = Profiler(interval=0.001)
prof.start()

t0 = time.perf_counter()
sim = WildfireSimulation(parameters=params, seed=3000)
sim.start()
# Block until the sim stops itself or we hit the wall budget.
deadline = t0 + WALL_BUDGET_S
while not sim.is_stopped.is_set():
    if time.perf_counter() > deadline:
        print(f"  wall budget hit for {scen}; stopping sim")
        try:
            sim.stop()
        except Exception:
            pass
        break
    sim.is_stopped.wait(timeout=1.0)
sim.join(timeout=10.0)
elapsed = time.perf_counter() - t0

prof.stop()
html_path = outdir / f"{tag}.html"
txt_path  = outdir / f"{tag}.txt"
html_path.write_text(prof.output_html())
txt_path.write_text(prof.output_text(unicode=True, color=False, show_all=False))
print(f"  wrote {html_path}")
print(f"  wrote {txt_path}")
print(f"  episode wall-time: {elapsed:.1f}s")
PYEOF
done

echo "============================================================"
echo ">>> DONE. Outputs in $OUTDIR_BASE"
echo "============================================================"
ls -la "$OUTDIR_BASE"
