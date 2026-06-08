#!/usr/bin/env bash
#
# Follow-up to profile_scenario_runtimes.slurm (job 500674).
# That run hit the 4-hour wall limit before reaching the switch-scenario
# and pyinstrument phases. Solo profile data for Palisades copy.json and
# Pyrenees.json is already complete in
#   examples/wildfire/data/scenarios/outputs/profile_500674/
# (Salamis got 2 of 6 rollouts, which is enough for stable means.)
#
# This script picks up the missing phases:
#   1) Switch-scenario run (round-robin Palisades/Pyrenees/Salamis in one
#      Python process). Apples-to-apples per-rollout summary table under
#      identical OS/cache state.
#   2) pyinstrument trace of a single headless episode per scenario, with
#      a strict per-scenario wall budget so a runaway Pyrenees / Salamis
#      mega-fire cannot blow the SLURM wall limit.
#
#SBATCH -J profile-rl-fu
#SBATCH -p normal_q
#SBATCH -A stselab_sos
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH -t 06:00:00
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

NUM_ENVS=12

# Switch phase: 4 rollouts × 144 n_steps × 12 envs = 6912 steps. Slowest
# worker step (Salamis ~2.8 s / Pyrenees ~2.7 s) dominates SubprocVecEnv
# sync, so 4 rollouts ≈ 4 × (~400 s step + reset overhead) ≈ 30-45 min
# wall. Keep small so we leave time for pyinstrument and a margin.
SWITCH_TIMESTEPS=6912

OUTDIR_BASE=examples/wildfire/data/scenarios/outputs/profile_followup_$SLURM_JOB_ID
mkdir -p "$OUTDIR_BASE"

echo "============================================================"
echo ">>> SWITCH-SCENARIO PROFILE (round-robin)"
echo "============================================================"
/usr/bin/time -v python -u -m examples.wildfire.ppo_runner \
  --switch-scenario \
  --switch-scenarios "Palisades copy.json" "Pyrenees.json" "Salamis.json" \
  --switch-ignition-1 \
  --num-envs "$NUM_ENVS" \
  --timesteps "$SWITCH_TIMESTEPS" \
  --device cpu \
  --profile-timings \
  --gc-collect-on-reset \
  --output-dir "$OUTDIR_BASE/switch_all" \
  --progress-file-tag "profile_switch_followup" \
  --vec-start-method auto

echo "============================================================"
echo ">>> PYINSTRUMENT PER-SCENARIO HEADLESS RUN"
echo "============================================================"

PYPROFILE_DIR="$OUTDIR_BASE/pyinstrument"
mkdir -p "$PYPROFILE_DIR"
export PYPROFILE_DIR

# Strict per-scenario wall budget. The fire model can run for hours of
# simulated time, so we cap by wall-clock to guarantee progress.
export PYINSTR_WALL_BUDGET_S=900

for SCENARIO in "Palisades copy.json" "Pyrenees.json" "Salamis.json"; do
  TAG=$(echo "$SCENARIO" | sed 's/\.json$//' | tr '[:upper:] ' '[:lower:]_')
  echo "--- pyinstrument: $SCENARIO (wall budget ${PYINSTR_WALL_BUDGET_S}s) ---"
  SCEN="$SCENARIO" TAG="$TAG" python -u - <<'PYEOF'
import os, time
from pathlib import Path
from pyinstrument import Profiler

from examples.wildfire.paths import SCENARIOS_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation

scen   = os.environ["SCEN"]
tag    = os.environ["TAG"]
outdir = Path(os.environ["PYPROFILE_DIR"])
budget = float(os.environ.get("PYINSTR_WALL_BUDGET_S", "900"))
outdir.mkdir(parents=True, exist_ok=True)

scenario_path = SCENARIOS_DIR / "inputs" / scen
print(f"Profiling headless episode for {scen} (budget {budget:.0f}s)")
with scenario_path.open() as f:
    params = WildfireParameters.model_validate_json(f.read())

prof = Profiler(interval=0.001)
prof.start()
t0 = time.perf_counter()

sim = WildfireSimulation(parameters=params, seed=3000)
sim.start()
deadline = t0 + budget
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
print(f"  episode wall-time: {elapsed:.1f}s")
print(f"  wrote {html_path}")
print(f"  wrote {txt_path}")
PYEOF
done

echo "============================================================"
echo ">>> DONE. Outputs under $OUTDIR_BASE"
echo "============================================================"
ls -la "$OUTDIR_BASE"
ls -la "$OUTDIR_BASE/pyinstrument" 2>/dev/null || true
